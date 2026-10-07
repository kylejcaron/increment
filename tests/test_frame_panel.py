"""Tests for the dataframe entry point (``increment/frame.py``).

The load-bearing tests are ``test_moments_match_group_summary`` and
``test_daily_moments_match_daily_group_summary``: they assert the narwhals
moments equal the real ibis builders column for column. Both import ibis;
``increment/frame.py`` must not.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from typing import Any, cast

import narwhals as nw
import pyarrow as pa
import pytest

from increment import readouts
from increment._frame_panel import _day_axis_label_order
from increment.errors import CapabilityError, IncrementWarning, InvalidRequestError
from increment.estimation.diagnostics import SRMResult
from increment.frame import (
    FramePanelSource,
    FrameTotalsSource,
    MetricSpec,
    from_unit_panel,
    from_unit_summary,
    synthesise_metric,
)
from increment.semantics.design import Randomized
from increment.semantics.models import Metric
from tests.warning_codes import warning_codes

_DESIGN = Randomized(
    control_group="control",
    allocation={"control": 0.5, "treatment": 0.5},
)


_CANCELLING_REFS = {"cy1": "ref_y", "cx1": "ref_x", "cden1": "ref_den"}


def _assert_moments_agree(got: Any, exp: Any, cols: tuple[str, ...]) -> None:
    """Assert two centered-moment rows agree on *cols* at rel=1e-9.

    Cancelling first moments get an absolute tolerance scaled to n*|ref|,
    which pins the recovered raw sum n*ref + c1 at that same 1e-9.
    """
    for col in cols:
        if col in _CANCELLING_REFS:
            scale = 1e-9 * max(1.0, exp["n"] * abs(exp[_CANCELLING_REFS[col]]))
            assert got[col] == pytest.approx(exp[col], abs=scale), col
        else:
            assert got[col] == pytest.approx(exp[col], rel=1e-9), col


def _sum_y(row: Any) -> float:
    """Recover sum(y) from a moments row: n*ref_y + cy1, exact and
    cancellation-free."""
    return row["n"] * row["ref_y"] + row["cy1"]


def _sum_den(row: Any) -> float:
    """Recover sum(den) from a moments row: n*ref_den + cden1."""
    return row["n"] * row["ref_den"] + row["cden1"]


def _metric(src: FrameTotalsSource | FramePanelSource, name: str) -> Metric:
    """Typed lookup into ``src.metrics`` - the protocol types it
    ``Sequence[object]`` (zero-dependency seam), so tests that need
    ``.name`` narrow it back explicitly rather than fighting ty at every
    call site."""
    return next(m for m in src.context.metrics if m.name == name)


def _reported_missing(exc: InvalidRequestError) -> set[str]:
    """Column names in a missing-column refusal's structured payload.

    ``context["missing"]`` carries one ``"<column>" (<role>)`` entry per
    column not found in the frame.
    """
    entries = exc.context["missing"]
    assert isinstance(entries, list | tuple)
    return {str(entry).split("'")[1] for entry in entries}


_PANEL_ROWS = [
    # unit, variant,     day,   revenue, orders
    ("u1", "control", "d1", 4.0, 2.0),
    ("u1", "control", "d2", 8.0, 3.0),
    ("u1", "control", "d3", 0.0, 0.0),
    ("u2", "control", "d1", 0.0, 0.0),
    ("u2", "control", "d2", 3.0, 1.0),
    ("u2", "control", "d3", 6.0, 2.0),
    ("u3", "control", "d1", 2.0, 1.0),
    ("u3", "control", "d2", 2.0, 1.0),
    ("u3", "control", "d3", 2.0, 1.0),
    ("u4", "control", "d1", 10.0, 4.0),
    ("u4", "control", "d2", 0.0, 0.0),
    ("u4", "control", "d3", 5.0, 2.0),
    ("u5", "treatment", "d1", 6.0, 3.0),
    ("u5", "treatment", "d2", 6.0, 3.0),
    ("u5", "treatment", "d3", 6.0, 3.0),
    ("u6", "treatment", "d1", 12.0, 5.0),
    ("u6", "treatment", "d2", 0.0, 0.0),
    ("u6", "treatment", "d3", 3.0, 1.0),
    ("u7", "treatment", "d1", 4.0, 2.0),
    ("u7", "treatment", "d2", 9.0, 4.0),
    ("u7", "treatment", "d3", 4.0, 2.0),
    ("u8", "treatment", "d1", 0.0, 0.0),
    ("u8", "treatment", "d2", 5.0, 2.0),
    ("u8", "treatment", "d3", 8.0, 3.0),
]


_PANEL_COLUMNS = ["user_id", "variant", "day", "revenue", "orders"]


def _panel_table(rows=None) -> pa.Table:
    rows = _PANEL_ROWS if rows is None else rows
    cols = list(zip(*rows, strict=True))
    return pa.table(dict(zip(_PANEL_COLUMNS, cols, strict=True)))


def _panel_with_breakouts() -> pa.Table:
    return pa.table(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"],
            "variant": ["control"] * 4 + ["treatment"] * 4,
            "day": ["d1", "d2"] * 4,
            "country": ["US", "US", "CA", "CA", "US", "US", "CA", "CA"],
            "plan": ["free", "free", "paid", "paid", "free", "free", "paid", "paid"],
            "revenue": [1.0, 2.0, 3.0, 5.0, 2.0, 4.0, 6.0, 9.0],
        }
    )


@pytest.fixture
def panel_frame() -> pa.Table:
    return _panel_table()


def test_parallel_panel_path_remains_independent_of_switchback_validation(
    panel_frame: pa.Table,
) -> None:
    """Importing the switchback constructor must not change parallel panel semantics."""
    source = from_unit_panel(
        panel_frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )
    assert source.unit_counts() == {"control": 4, "treatment": 4}


def test_daily_moments_match_daily_group_summary(panel_frame: pa.Table) -> None:
    """The narwhals per-day moments equal builders.daily_group_summary, all columns.

    Parity with the ibis builders for the panel path, mirroring
    ``test_moments_match_group_summary`` for the summary path.
    """
    ibis = pytest.importorskip("ibis")
    pytest.importorskip("duckdb")
    from increment.query.builders import daily_group_summary

    values = [r[3] for r in _PANEL_ROWS]
    unit_day_panel_rows = {
        "unit_id": [r[0] for r in _PANEL_ROWS],
        "ds": [r[2] for r in _PANEL_ROWS],
        "experiment_id": ["frame"] * len(_PANEL_ROWS),
        "group_id": [r[1] for r in _PANEL_ROWS],
        "metric": ["revenue"] * len(_PANEL_ROWS),
        "n_events": [1] * len(_PANEL_ROWS),
        "sum_value": values,
        "min_value": values,
        "max_value": values,
    }
    con = ibis.duckdb.connect()
    panel = ibis.memtable(unit_day_panel_rows)
    metric = synthesise_metric(MetricSpec(name="revenue", type="mean"))
    expected = con.to_pyarrow(daily_group_summary(panel, metric=metric)).to_pylist()

    analysis = from_unit_panel(
        panel_frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )
    assert analysis.densified_cells == 0, "fixture is dense by construction"
    actual = {(r["ds"], r["group_id"]): r for r in readouts.daily(analysis)}

    assert len(expected) == len(actual)
    for exp in expected:
        got = actual[
            (
                exp["ds"].isoformat() if hasattr(exp["ds"], "isoformat") else exp["ds"],
                exp["group_id"],
            )
        ]
        assert got["n"] == exp["n"]
        _assert_moments_agree(got, exp, ("ref_y", "cy1", "cy2"))
        for col in ("ref_x", "cx1", "cx2", "cxy", "ref_den", "cden1", "cden2", "cyden"):
            assert got[col] is None and exp[col] is None, col

    # Ratio metric: ref_den/cden1/cden2/cyden are populated via a second dense
    # panel (orders) joined at (unit_id, ds); the mean-metric check above never exercises it since it stays null there by design.
    den_values = [r[4] for r in _PANEL_ROWS]
    den_panel_rows = {
        **unit_day_panel_rows,
        "metric": ["aov"] * len(_PANEL_ROWS),
        "sum_value": den_values,
        "min_value": den_values,
        "max_value": den_values,
    }
    den_panel = ibis.memtable(den_panel_rows)
    ratio_spec = MetricSpec(name="aov", type="ratio", numerator="revenue", denominator="orders")
    expected_ratio = con.to_pyarrow(
        daily_group_summary(
            panel.mutate(metric=ibis.literal("aov")),
            metric=synthesise_metric(ratio_spec),
            den_panel=den_panel,
        )
    ).to_pylist()

    ratio_analysis = from_unit_panel(
        panel_frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[ratio_spec],
    )
    actual_ratio = {(r["ds"], r["group_id"]): r for r in readouts.daily(ratio_analysis)}

    assert len(expected_ratio) == len(actual_ratio)
    for exp in expected_ratio:
        got = actual_ratio[
            (
                exp["ds"].isoformat() if hasattr(exp["ds"], "isoformat") else exp["ds"],
                exp["group_id"],
            )
        ]
        assert got["n"] == exp["n"]
        _assert_moments_agree(
            got, exp, ("ref_y", "cy1", "cy2", "ref_den", "cden1", "cden2", "cyden")
        )
        for col in ("ref_x", "cx1", "cx2", "cxy"):
            assert got[col] is None and exp[col] is None, col


def test_asof_moments_gate_units_until_declared_exposure_date() -> None:
    base = date(2026, 1, 1)
    rows = [
        ("c1", "control", base, base, "US", 1.0),
        ("c1", "control", base + timedelta(days=1), base, "US", 1.0),
        ("t1", "treatment", base, base + timedelta(days=1), "US", 9.0),
        ("t1", "treatment", base + timedelta(days=1), base + timedelta(days=1), "US", 2.0),
    ]
    frame = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "exposed_on", "country", "revenue"],
                zip(*rows, strict=True),
                strict=True,
            )
        )
    )
    src = from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        exposure_date="exposed_on",
        breakouts=["country"],
    )

    moments = src.moments(_metric(src, "revenue"), grain="asof", by=["country"])
    day1 = [row for row in moments if row["ds"] == base]
    assert {(row["group_id"], row["n"]) for row in day1} == {("control", 1)}


def test_asof_moments_without_exposure_uses_fixed_global_cohort() -> None:
    base = date(2026, 1, 1)
    rows = [
        ("c1", "control", base, 1.0),
        ("c1", "control", base + timedelta(days=1), 1.0),
        ("t1", "treatment", base, 0.0),
        ("t1", "treatment", base + timedelta(days=1), 2.0),
    ]
    frame = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "revenue"],
                zip(*rows, strict=True),
                strict=True,
            )
        )
    )
    src = from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )

    moments = src.moments(_metric(src, "revenue"), grain="asof")
    day1 = [row for row in moments if row["ds"] == base]
    assert {(row["group_id"], row["n"]) for row in day1} == {
        ("control", 1),
        ("treatment", 1),
    }


def test_asof_cumulative_moments_order_string_day_labels_naturally() -> None:
    """String day labels sharing one consistent prefix + signed-integer
    shape (e.g. "d1"/"d2"/"d10") order by the parsed integer, not raw
    string comparison - previously producing cumulative 1, 11, 13 instead
    of the chronological 1, 3, 13."""
    rows = [
        ("u1", "control", "d1", 1.0),
        ("u1", "control", "d10", 10.0),
        ("u1", "control", "d2", 2.0),
    ]
    frame = pa.table(
        dict(zip(["user_id", "variant", "day", "revenue"], zip(*rows, strict=True), strict=True))
    )
    src = from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )
    moments = src.moments(_metric(src, "revenue"), grain="asof")
    cumulative = {row["ds"]: _sum_y(row) for row in moments}
    assert cumulative == {"d1": 1.0, "d2": 3.0, "d10": 13.0}


def test_asof_cumulative_moments_refuses_ambiguous_string_day_labels() -> None:
    """A "YYYY-M" year-month label is neither an unambiguous ISO-8601 date
    nor a single consistent prefix + signed-integer shape - raw
    lexicographic sort gets both the within-year digit-width jump and the
    year rollover wrong, and a digit-chunking "natural sort" guess is
    exactly as unsound; refuses instead of silently guessing an order."""
    rows = [
        ("u1", "control", "2025-9", 1.0),
        ("u1", "control", "2025-10", 2.0),
        ("u1", "control", "2026-1", 4.0),
    ]
    frame = pa.table(
        dict(zip(["user_id", "variant", "day", "revenue"], zip(*rows, strict=True), strict=True))
    )
    src = from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )
    with pytest.raises(CapabilityError) as raised:
        src.moments(_metric(src, "revenue"), grain="asof")
    assert raised.value.code == "frame.asof.day_axis_unorderable"


def test_asof_encouragement_uptake_moments_match_group_summary() -> None:
    ibis = pytest.importorskip("ibis")
    pytest.importorskip("duckdb")
    import pandas as pd

    from increment.query.builders import asof_group_summary
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    base = date(2026, 3, 1)
    units = [
        ("c1", "control", 0),
        ("c2", "control", 1),
        ("t1", "treatment", 0),
        ("t2", "treatment", 1),
    ]
    rows: list[dict[str, Any]] = []
    outcome_panel: list[dict[str, Any]] = []
    uptake_panel: list[dict[str, Any]] = []
    for unit_index, (unit_id, group_id, exposure_offset) in enumerate(units):
        exposure = base + timedelta(days=exposure_offset)
        for elapsed in range(5 - exposure_offset):
            ds = exposure + timedelta(days=elapsed)
            clicked = float(
                (unit_id == "c1" and elapsed == 1)
                or (unit_id == "c2" and elapsed in (2, 3))
                or (unit_id == "t1" and elapsed == 3)
                or (unit_id == "t2" and elapsed == 1)
            )
            value = float(unit_index + elapsed + 1)
            rows.append(
                {
                    "user_id": unit_id,
                    "variant": group_id,
                    "day": ds,
                    "exposed_on": exposure,
                    "revenue": value,
                    "clicked": clicked,
                }
            )
            base_row = {
                "unit_id": unit_id,
                "ds": ds,
                "experiment_id": "frame",
                "metric": "revenue",
                "group_id": group_id,
                "first_exposure_date": exposure,
            }
            outcome_panel.append(
                {
                    **base_row,
                    "n_events": 1,
                    "sum_value": value,
                    "min_value": value,
                    "max_value": value,
                }
            )
            uptake_panel.append(
                {
                    **base_row,
                    "n_events": 1,
                    "sum_value": clicked,
                    "min_value": clicked,
                    "max_value": clicked,
                }
            )

    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked", window_days=3),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
    )
    metric = synthesise_metric(MetricSpec(name="revenue", type="mean"))
    con = ibis.duckdb.connect()
    expected = {
        (row["ds"], row["group_id"]): row
        for row in con.to_pyarrow(
            asof_group_summary(
                ibis.memtable(pd.DataFrame(outcome_panel)),
                metric,
                uptake_panel=ibis.memtable(pd.DataFrame(uptake_panel)),
                uptake_window_days=3,
            )
        ).to_pylist()
    }
    src = from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        exposure_date="exposed_on",
        uptake="clicked",
        design=design,
    )
    actual = {
        (row["ds"], row["group_id"]): row
        for row in src.moments(_metric(src, "revenue"), grain="asof")
    }

    assert actual.keys() == expected.keys()
    for key, expected_row in expected.items():
        actual_row = actual[key]
        assert actual_row["sum_d"] == pytest.approx(expected_row["sum_d"])
        _assert_moments_agree(actual_row, expected_row, ("cyd", "cy2d"))
        assert actual_row["cxd"] is None


def test_asof_uptake_window_without_exposure_uses_first_observed_numeric_day() -> None:
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    rows = [
        ("c1", "control", 0, 1.0, 0.0),
        ("c1", "control", 1, 1.0, 0.0),
        ("c1", "control", 2, 1.0, 0.0),
        ("c1", "control", 3, 1.0, 0.0),
        ("c1", "control", 4, 1.0, 0.0),
        ("c1", "control", 5, 1.0, 0.0),
        ("t1", "treatment", 0, 2.0, 0.0),
        ("t1", "treatment", 1, 2.0, 0.0),
        ("t1", "treatment", 2, 2.0, 0.0),
        ("t1", "treatment", 3, 2.0, 0.0),
        ("t1", "treatment", 4, 2.0, 0.0),
        ("t1", "treatment", 5, 2.0, 0.0),
        ("t2", "treatment", 2, 3.0, 0.0),
        ("t2", "treatment", 3, 3.0, 0.0),
        ("t2", "treatment", 4, 3.0, 1.0),
        ("t2", "treatment", 5, 3.0, 1.0),
    ]
    src = from_unit_panel(
        pa.table(
            dict(
                zip(
                    ["user_id", "variant", "day", "revenue", "clicked"],
                    zip(*rows, strict=True),
                    strict=True,
                )
            )
        ),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        uptake="clicked",
        design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked", window_days=3),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True, justification="assignment only moves revenue via uptake"
            ),
        ),
    )

    moments = src.moments(_metric(src, "revenue"), grain="asof")
    by_day_group = {(row["ds"], row["group_id"]): row for row in moments}
    assert by_day_group[(0, "treatment")]["n"] == 2
    final_treatment = by_day_group[(5, "treatment")]
    assert final_treatment["sum_d"] == 1.0
    assert final_treatment["cyd"] is not None
    assert final_treatment["cy2d"] is not None
    assert final_treatment["cxd"] is None


def test_windowed_metric_numeric_day_axis_respects_window_edge() -> None:
    """A numeric day axis must be measured in whole days directly, not
    cast through Datetime('us') as microseconds - regression: ds=[0,1],
    window_days=1 previously admitted both days (day 1 read as
    ~1.16e-11 elapsed days) instead of only day 0."""
    rows = [
        ("c1", "control", 0, 0, 1.0),
        ("c1", "control", 1, 0, 2.0),
        ("t1", "treatment", 0, 0, 3.0),
        ("t1", "treatment", 1, 0, 4.0),
    ]
    frame = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "exposed_on", "revenue"],
                zip(*rows, strict=True),
                strict=True,
            )
        )
    )
    src = from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", window_days=1)],
        exposure_date="exposed_on",
    )
    moments = {r["group_id"]: r for r in src.moments(_metric(src, "revenue"), grain="total")}
    assert _sum_y(moments["control"]) == pytest.approx(1.0)
    assert _sum_y(moments["treatment"]) == pytest.approx(3.0)


def test_numeric_day_axis_with_calendar_exposure_date_refuses() -> None:
    """A numeric date column combined with a calendar exposure_date is a
    contradictory day axis - previously silently filtered every row as
    pre-exposure (moments == []) while unit_counts() still reported both
    arms; now refused AT CONSTRUCTION, naming both columns, so an
    inconsistent source can never be built at all."""
    base = date(2026, 1, 1)
    rows = [
        ("c1", "control", 0, base, 1.0),
        ("c1", "control", 1, base, 2.0),
    ]
    frame = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "exposed_on", "revenue"],
                zip(*rows, strict=True),
                strict=True,
            )
        )
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_panel(
            frame,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[MetricSpec(name="revenue", window_days=1)],
            exposure_date="exposed_on",
        )
    assert exc_info.value.code == "frame.validation.dtype_disagree_day"


def test_asof_cumulative_moments_refuses_ambiguous_date_like_labels() -> None:
    """A "MM/DD/YYYY" label is not an unambiguous ISO-8601 date - raw
    lexicographic sort would read "01/01/2026" as before "12/31/2025"
    (comparing the leading '0' against '1'); refuses rather than
    silently mis-ordering."""
    rows = [
        ("u1", "control", "12/31/2025", 1.0),
        ("u1", "control", "01/01/2026", 2.0),
    ]
    frame = pa.table(
        dict(zip(["user_id", "variant", "day", "revenue"], zip(*rows, strict=True), strict=True))
    )
    src = from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )
    with pytest.raises(CapabilityError) as raised:
        src.moments(_metric(src, "revenue"), grain="asof")
    assert raised.value.code == "frame.asof.day_axis_unorderable"


def test_asof_cumulative_moments_orders_negative_prefixed_integer_labels() -> None:
    """ "d-1" (day -1) must sort before "d0" (day 0) by parsed integer -
    the old digit-chunking natural-sort heuristic got this backwards
    (``('d', 0, '') < ('d-', 1, '')``, i.e. "d0" before "d-1")."""
    rows = [
        ("u1", "control", "d0", 1.0),
        ("u1", "control", "d-1", 2.0),
    ]
    frame = pa.table(
        dict(zip(["user_id", "variant", "day", "revenue"], zip(*rows, strict=True), strict=True))
    )
    src = from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )
    moments = src.moments(_metric(src, "revenue"), grain="asof")
    cumulative = {row["ds"]: _sum_y(row) for row in moments}
    assert cumulative == {"d-1": 2.0, "d0": 3.0}


def test_asof_cumulative_moments_refuses_colliding_prefixed_integer_labels() -> None:
    """ "d1" and "d01" parse to the same integer under the consistent
    "<prefix><signed integer>" rule; admitting both would silently make
    the cumulative order (and its running sum) depend on backend/input
    row order. Refuses instead, naming the colliding labels."""
    rows = [
        ("u1", "control", "d1", 1.0),
        ("u1", "control", "d01", 2.0),
    ]
    frame = pa.table(
        dict(zip(["user_id", "variant", "day", "revenue"], zip(*rows, strict=True), strict=True))
    )
    src = from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )
    with pytest.raises(CapabilityError) as raised:
        src.moments(_metric(src, "revenue"), grain="asof")
    assert raised.value.code == "frame.asof.day_axis_unorderable"


def test_asof_cumulative_moments_invariant_to_input_row_order() -> None:
    """Aggregate calculations must be invariant to input row order (repo
    convention): shuffling the input rows for an accepted day-label set
    must not change the emitted cumulative as-of moments."""
    import random

    rows = [
        ("u1", "control", "d1", 1.0),
        ("u1", "control", "d2", 2.0),
        ("u1", "control", "d3", 3.0),
        ("u2", "treatment", "d1", 4.0),
        ("u2", "treatment", "d2", 5.0),
        ("u2", "treatment", "d3", 6.0),
    ]

    def build(order: list[int]) -> list[dict[str, Any]]:
        ordered = [rows[i] for i in order]
        frame = pa.table(
            dict(
                zip(
                    ["user_id", "variant", "day", "revenue"],
                    zip(*ordered, strict=True),
                    strict=True,
                )
            )
        )
        src = from_unit_panel(
            frame,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
        )
        return src.moments(_metric(src, "revenue"), grain="asof")

    def normalize(moments: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
        return {(row["ds"], row["group_id"], _sum_y(row), row["n"]) for row in moments}

    baseline = normalize(build(list(range(len(rows)))))
    shuffled_order = list(range(len(rows)))
    random.Random(0).shuffle(shuffled_order)
    shuffled = normalize(build(shuffled_order))
    assert baseline == shuffled


def test_observable_end_kind_mismatch_refuses() -> None:
    """A numeric day axis with a calendar observation_end reaches
    unsupported date-minus-number backend arithmetic; refused instead,
    naming both kinds."""
    rows = [
        ("c1", "control", 0, 0, 1.0),
        ("c1", "control", 1, 0, 2.0),
    ]
    frame = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "exposed_on", "revenue"],
                zip(*rows, strict=True),
                strict=True,
            )
        )
    )
    src = from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", window_days=1)],
        exposure_date="exposed_on",
        observation_end=date(2026, 1, 1),
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        src.moments(_metric(src, "revenue"), grain="total")
    assert exc_info.value.code == "frame.panel.observable_end_dtype"
    assert exc_info.value.context["axis_kind"] == "numeric"
    assert exc_info.value.context["end_kind"] == "calendar"


def test_infinite_date_on_numeric_axis_refuses() -> None:
    """An infinite (+-inf) value on a numeric day axis silently distorts
    window maturity math and day-index filtering; refused at
    construction alongside the null/NaN date check."""
    rows = [
        ("c1", "control", 0.0, 1.0),
        ("c1", "control", float("inf"), 2.0),
    ]
    frame = pa.table(
        dict(zip(["user_id", "variant", "day", "revenue"], zip(*rows, strict=True), strict=True))
    )
    with pytest.raises(InvalidRequestError) as exc:
        from_unit_panel(
            frame,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
        )
    assert exc.value.code == "source.frame.non_finite"


def test_sparse_panel_daily_moments_include_inactive_units_as_zero() -> None:
    """Logical zero population preserves the measured 2.333-vs-7.000 result.

    Three units, one inactive on day 2: n=3 and mean 7/3; omitting the logical
    zero would yield n=1 and mean 7.0.
    """
    sparse_rows = [
        ("u1", "control", "d1", 1.0, 1.0),
        ("u1", "control", "d2", 7.0, 1.0),
        ("u1", "control", "d3", 1.0, 1.0),
        ("u2", "control", "d1", 1.0, 1.0),
        # u2 has no row on d2 - inactive that day.
        ("u2", "control", "d3", 1.0, 1.0),
        ("u3", "control", "d1", 1.0, 1.0),
        ("u3", "control", "d2", 0.0, 1.0),
        ("u3", "control", "d3", 1.0, 1.0),
    ]
    analysis = from_unit_panel(
        _panel_table(sparse_rows),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )
    assert analysis.densified_cells == 1

    day2 = next(r for r in readouts.daily(analysis) if r["ds"] == "d2")
    assert day2["n"] == 3
    # The wire carries the first moment as n*ref_y + cy1, exact by design.
    assert day2["n"] * day2["ref_y"] + day2["cy1"] == pytest.approx(7.0)
    assert day2["ref_y"] == pytest.approx(2.333, abs=1e-3)
    assert day2["ref_y"] != pytest.approx(7.0)


def _assert_sparse_panel_moments_match(
    actual_rows: Mapping[tuple[Any, ...], dict[str, Any]],
    oracle: Mapping[tuple[Any, ...], dict[str, Any]],
    scenario: str,
) -> None:
    import math

    assert actual_rows.keys() == oracle.keys()
    for key, row in actual_rows.items():
        expected = oracle[key]
        assert row.keys() == expected.keys()
        for name, value in row.items():
            wanted = expected[name]
            if isinstance(value, float):
                assert math.isfinite(value), (scenario, key, name, value)
                assert value == pytest.approx(wanted, rel=1e-9, abs=1e-12), (
                    scenario,
                    key,
                    name,
                )
            else:
                assert value == wanted, (scenario, key, name)


def _sparse_panel_oracle_fixture(
    scenario: str,
) -> tuple[list[date], list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np

    days = [date(2026, 1, day) for day in range(1, 7)]
    sparse_rows: list[dict[str, Any]] = []
    dense_rows: list[dict[str, Any]] = []
    low = np.float64(1e150)
    high = np.nextafter(low, np.inf)
    tiny = np.float64(1e-150)
    tiny_high = np.nextafter(tiny, np.inf)
    for unit_index in range(8):
        unit = f"u{unit_index}"
        arm = "control" if unit_index < 4 else "treatment"
        country = "CA" if unit_index % 2 == 0 else "US"
        exposure_index = 0 if scenario == "precision_order" else unit_index % 2
        exposure = days[exposure_index]
        observed: dict[date, dict[str, Any]] = {}
        for day_index, day in enumerate(days):
            before_exposure = day_index == exposure_index - 1
            boundary = day_index == exposure_index + 3
            after_boundary = day_index == exposure_index + 4
            window_days = 4 if scenario == "precision_order" else 3
            in_window = exposure_index <= day_index < exposure_index + window_days
            global_last_date = day_index == len(days) - 1
            should_observe = (
                before_exposure
                or boundary
                or after_boundary
                or global_last_date
                or (
                    in_window
                    and (
                        scenario == "precision_order"
                        or (unit_index + day_index) % 3 != 0
                        or day_index == exposure_index
                    )
                )
            )
            if not should_observe:
                continue
            precision_values = (
                (1e16, 1.0, -1e16, 1.0) if unit_index < 4 else (1e16, 1.0, -1e16, 2.0)
            )
            observed[day] = {
                "numerator": float(low if (unit_index + day_index) % 2 == 0 else high),
                "denominator": float(tiny if (unit_index + day_index) % 2 == 0 else tiny_high),
                "retained": float(day_index in (exposure_index - 1, exposure_index + 1)),
                "outcome": float(day_index in (exposure_index, exposure_index + 3)),
                "clicked": float(day_index in (exposure_index, exposure_index + 3)),
                "precision": (
                    precision_values[day_index - exposure_index]
                    if 0 <= day_index - exposure_index < len(precision_values)
                    else 0.0
                ),
            }
        for day in days:
            values = observed.get(
                day,
                {
                    "numerator": 0.0,
                    "denominator": 0.0,
                    "retained": 0.0,
                    "outcome": 0.0,
                    "clicked": 0.0,
                    "precision": 0.0,
                },
            )
            row = {
                "unit": unit,
                "arm": arm,
                "country": country,
                "day": day,
                "exposed": exposure,
                **values,
            }
            dense_rows.append(row)
            if day in observed:
                sparse_rows.append(row)
    return days, sparse_rows, dense_rows


@pytest.mark.parametrize("scenario", ["windowed_ratio", "retention", "uptake", "precision_order"])
def test_sparse_panel_oracle_matches_dense_across_execution_kernels(
    scenario: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare public sparse and explicit-zero populations across cutover, gates, and row order."""

    import increment.frame as frame_module
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    days, sparse_rows, dense_rows = _sparse_panel_oracle_fixture(scenario)
    if scenario == "windowed_ratio":
        specs = [
            MetricSpec(
                name="ratio",
                type="ratio",
                numerator="numerator",
                denominator="denominator",
                window_days=3,
            )
        ]
        uptake = None
        design = None
        grains = ("daily", "asof")
    elif scenario == "retention":
        specs = [
            MetricSpec(
                name="retention",
                type="retention",
                value_column="retained",
                threshold_days=(0, 2),
            )
        ]
        uptake = None
        design = None
        grains = ("asof",)
    elif scenario == "precision_order":
        specs = [MetricSpec(name="precision", type="mean", value_column="precision", window_days=4)]
        uptake = None
        design = None
        grains = ("asof",)
    else:
        specs = [MetricSpec(name="outcome", type="mean", value_column="outcome", window_days=3)]
        uptake = "clicked"
        design = Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked", window_days=3),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True, justification="assignment affects outcome only through uptake"
            ),
        )
        grains = ("daily", "asof")

    breakouts = [] if scenario == "precision_order" else ["country"]

    def build(rows: list[dict[str, Any]], *, layout: str) -> FramePanelSource:
        if layout == "unchunked":
            panel = pa.Table.from_pylist(rows)
        else:
            day_slices = (days[:2], days[2:]) if layout == "forward" else (days[2:], days[:2])
            panel = pa.concat_tables(
                [
                    pa.Table.from_pylist([row for row in rows if row["day"] in day_slice])
                    for day_slice in day_slices
                ]
            )
        return from_unit_panel(
            panel,
            unit="unit",
            group="arm",
            date="day",
            exposure_date="exposed",
            control="control",
            metrics=specs,
            breakouts=breakouts,
            uptake=uptake,
            design=design,
            observation_end=days[-1],
        )

    def read(
        rows: list[dict[str, Any]], budget: int, *, layout: str
    ) -> dict[tuple[Any, ...], dict[str, Any]]:
        monkeypatch.setattr(frame_module, "_DAY_PANEL_SCRATCH_BUDGET_BYTES", budget)
        source = build(rows, layout=layout)
        metric = _metric(source, specs[0].name)
        result: dict[tuple[Any, ...], dict[str, Any]] = {}
        for grain in grains:
            for completed in (False, True):
                actual = source.moments(
                    metric,
                    grain=grain,
                    by=breakouts,
                    completed_windows_only=completed,
                )
                result.update(
                    {
                        (
                            grain,
                            completed,
                            row["ds"],
                            row["group_id"],
                            *(row[name] for name in breakouts),
                        ): row
                        for row in actual
                    }
                )
        return result

    oracle = read(dense_rows, 64 * 1024 * 1024, layout="unchunked")
    chunked_oracle = read(dense_rows, 64 * 1024 * 1024, layout="forward")
    reversed_batch_oracle = read(dense_rows, 64 * 1024 * 1024, layout="reverse")
    vectorized_sparse = read(list(reversed(sparse_rows)), 64 * 1024 * 1024, layout="forward")
    streaming_sparse = read(sparse_rows, 0, layout="forward")
    streaming_reversed = read(list(reversed(sparse_rows)), 0, layout="reverse")
    _assert_sparse_panel_moments_match(chunked_oracle, oracle, f"{scenario}/chunked")
    _assert_sparse_panel_moments_match(
        reversed_batch_oracle, oracle, f"{scenario}/reversed batches"
    )
    _assert_sparse_panel_moments_match(vectorized_sparse, oracle, scenario)
    _assert_sparse_panel_moments_match(streaming_sparse, oracle, scenario)
    _assert_sparse_panel_moments_match(streaming_reversed, oracle, scenario)
    if scenario == "uptake":
        assert any(key[0] == "asof" and row["sum_d"] > 0.0 for key, row in streaming_sparse.items())
    if scenario == "retention":
        assert any(row["successes"] > 0 for row in streaming_sparse.values())


def test_sparse_daily_and_asof_match_explicit_zero_panel_in_any_input_order() -> None:
    days = [date(2026, 1, day) for day in range(1, 5)]
    sparse = [
        {"unit": unit, "arm": arm, "day": days[index], "value": value}
        for unit, arm, index, value in [
            ("c1", "control", 0, 2.0),
            ("c2", "control", 1, 4.0),
            ("t1", "treatment", 2, 6.0),
            ("t2", "treatment", 3, 8.0),
        ]
    ]
    dense = [
        {
            "unit": row["unit"],
            "arm": row["arm"],
            "day": day,
            "value": row["value"] if day == row["day"] else 0.0,
        }
        for row in sparse
        for day in days
    ]

    def source(rows):
        return from_unit_panel(
            pa.Table.from_pylist(rows),
            unit="unit",
            group="arm",
            date="day",
            control="control",
            metrics={"value": "mean"},
        )

    sparse_source = source(sparse)
    dense_source = source(list(reversed(dense)))
    assert sparse_source.densified_cells == 12
    assert dense_source.densified_cells == 0
    metric = _metric(sparse_source, "value")
    dense_metric = _metric(dense_source, "value")
    for grain in ("daily", "asof"):
        actual = sparse_source.moments(metric, grain=grain)
        expected = dense_source.moments(dense_metric, grain=grain)
        assert {(r["ds"], r["group_id"]) for r in actual} == {
            (r["ds"], r["group_id"]) for r in expected
        }
        by_key = {(r["ds"], r["group_id"]): r for r in expected}
        for row in actual:
            wanted = by_key[(row["ds"], row["group_id"])]
            for name, value in row.items():
                if isinstance(value, float):
                    assert value == pytest.approx(wanted[name], rel=1e-9, abs=1e-12)
                else:
                    assert value == wanted[name]


def test_panel_breakouts_preserve_declared_order() -> None:
    src = from_unit_panel(
        _panel_with_breakouts(),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=["country", "plan"],
    )
    assert src.breakouts == ("country", "plan")


def test_panel_breakout_declarations_preserve_unsegmented_moments() -> None:
    unsegmented = from_unit_panel(
        _panel_with_breakouts(),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )
    declared_breakouts = from_unit_panel(
        _panel_with_breakouts(),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=["country", "plan"],
    )

    assert unsegmented.moments(_metric(unsegmented, "revenue")) == declared_breakouts.moments(
        _metric(declared_breakouts, "revenue")
    )
    assert unsegmented.moments(
        _metric(unsegmented, "revenue"), grain="daily"
    ) == declared_breakouts.moments(_metric(declared_breakouts, "revenue"), grain="daily")


@pytest.mark.parametrize(
    ("breakouts", "code"),
    [
        (["country", "country"], "frame.validation.duplicate_breakout_column"),
        (["user_id"], "frame.validation.breakout_overlaps_column"),
        (["variant"], "frame.validation.breakout_overlaps_column"),
        (["day"], "frame.validation.breakout_overlaps_column"),
        (["revenue"], "frame.validation.breakout_overlaps_metric"),
        (["unit_id"], "frame.validation.breakout_reserved_internal"),
        (["_dy"], "frame.validation.breakout_reserved_internal"),
    ],
)
def test_panel_breakouts_reject_invalid_declarations(breakouts: list[str], code: str) -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_panel(
            _panel_with_breakouts(),
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
            breakouts=breakouts,
        )
    assert exc_info.value.code == code


def test_panel_breakout_missing_columns_share_existing_diagnostic() -> None:
    with pytest.raises(InvalidRequestError) as exc:
        from_unit_panel(
            _panel_with_breakouts(),
            unit="missing_unit",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
            breakouts=["missing_country", "missing_plan"],
        )
    assert exc.value.code == "source.frame.metric_missing"
    assert _reported_missing(exc.value) == {"missing_unit", "missing_country", "missing_plan"}


def test_panel_breakout_rejects_unit_with_changing_value() -> None:
    frame = _panel_with_breakouts().set_column(
        3,
        "country",
        pa.array(["US", "CA", "CA", "CA", "US", "US", "CA", "CA"]),
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_panel(
            frame,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
            breakouts=["country"],
        )
    assert exc_info.value.code == "frame.validation.breakout_unit_stable"
    assert exc_info.value.context["name"] == "country"


def test_panel_breakout_rejects_mixed_null_and_non_null_identity() -> None:
    frame = _panel_with_breakouts().set_column(
        3,
        "country",
        pa.array(["US", None, "CA", "CA", "US", "US", "CA", "CA"]),
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_panel(
            frame,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
            breakouts=["country"],
        )
    assert exc_info.value.code == "frame.validation.breakout_unit_stable"
    assert exc_info.value.context["count"] == 1


def test_panel_breakout_normalizes_all_null_identity_to_sentinel() -> None:
    frame = _panel_with_breakouts().set_column(3, "country", pa.array([None] * 8, type=pa.string()))
    src = from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=["country"],
    )
    assert {row["country"] for row in src.moments(_metric(src, "revenue"), by=["country"])} == {
        "__null__"
    }


@pytest.mark.parametrize("kind", ["pandas", "polars", "arrow"])
def test_panel_boolean_breakout_moments_match_the_canonical_string_frame(kind: str) -> None:
    """Boolean and null breakout labels read ``true``/``false``/``__null__`` on every backend."""
    from tests.test_sequential_registration_contract import typed_segment_frame

    labels = ["true", "true", "false", "false", "true", "true", "__null__", "__null__"]
    canonical = _panel_with_breakouts().set_column(3, "country", pa.array(labels)).to_pandas()
    typed = typed_segment_frame(canonical, kind, column="country")

    def read(frame: Any) -> dict[str, list[dict[str, Any]]]:
        src = from_unit_panel(
            frame,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
            breakouts=["country"],
        )
        metric = _metric(src, "revenue")
        return {
            grain: sorted(src.moments(metric, grain=cast("Any", grain), by=["country"]), key=repr)
            for grain in ("total", "daily")
        }

    observed = read(typed)
    assert observed == read(canonical)
    assert {row["country"] for row in observed["total"]} == {"true", "false", "__null__"}


def test_panel_daily_moments_preserve_breakout_through_sparse_densification() -> None:
    rows = [
        {"user_id": "c_us", "variant": "control", "day": "d1", "country": "US", "revenue": 2.0},
        {"user_id": "c_us", "variant": "control", "day": "d2", "country": "US", "revenue": 4.0},
        {"user_id": "t_us", "variant": "treatment", "day": "d1", "country": "US", "revenue": 6.0},
        {"user_id": "c_ca", "variant": "control", "day": "d1", "country": "CA", "revenue": 3.0},
        {"user_id": "t_ca", "variant": "treatment", "day": "d1", "country": "CA", "revenue": 9.0},
    ]
    src = from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=["country"],
    )
    day2 = {
        (row["country"], row["group_id"]): row
        for row in src.moments(_metric(src, "revenue"), grain="daily", by=["country"])
        if row["ds"] == "d2"
    }

    assert set(day2) == {
        ("US", "control"),
        ("US", "treatment"),
        ("CA", "control"),
        ("CA", "treatment"),
    }
    assert day2[("US", "treatment")]["n"] == 1
    assert day2[("US", "treatment")]["ref_y"] == 0.0
    assert day2[("CA", "control")]["ref_y"] == 0.0


def test_panel_total_moments_group_by_declared_breakout() -> None:
    src = from_unit_panel(
        _panel_with_breakouts(),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=["country"],
    )

    rows = src.moments(_metric(src, "revenue"), by=["country"])
    assert {(row["country"], row["group_id"]) for row in rows} == {
        ("US", "control"),
        ("US", "treatment"),
        ("CA", "control"),
        ("CA", "treatment"),
    }


def test_panel_breakout_all_null_is_visible_segment() -> None:
    frame = _panel_with_breakouts().set_column(3, "country", pa.array([None] * 8))
    src = from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=["country"],
    )

    rows = src.moments(_metric(src, "revenue"), grain="daily", by=["country"])
    assert {row["country"] for row in rows} == {"__null__"}


@pytest.mark.parametrize("by", [["unknown"], ["country", "plan"]])
def test_panel_moments_reject_invalid_breakout_requests(by: list[str]) -> None:
    src = from_unit_panel(
        _panel_with_breakouts(),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=["country", "plan"],
    )

    with pytest.raises(CapabilityError):
        src.moments(_metric(src, "revenue"), grain="daily", by=by)


def test_panel_exposure_excludes_pre_exposure_rows_from_total_and_daily() -> None:
    base = date(2026, 1, 1)
    src = from_unit_panel(
        pa.table(
            {
                "user_id": ["c1", "c1", "t1", "t1"],
                "variant": ["control", "control", "treatment", "treatment"],
                "day": [base, base + timedelta(days=1), base, base + timedelta(days=1)],
                "exposed_on": [base, base, base + timedelta(days=1), base + timedelta(days=1)],
                "revenue": [1.0, 1.0, 100.0, 2.0],
            }
        ),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        exposure_date="exposed_on",
    )
    metric = _metric(src, "revenue")

    total = {row["group_id"]: row for row in src.moments(metric)}
    assert total["treatment"]["ref_y"] == 2.0
    first_day = [row for row in src.moments(metric, grain="daily") if row["ds"] == base]
    assert {(row["group_id"], row["n"]) for row in first_day} == {("control", 1)}


def test_sparse_totals_preserve_observable_units_without_post_exposure_rows() -> None:
    source = from_unit_panel(
        pa.table(
            {
                "unit": ["c0", "c1", "t0", "t1"],
                "arm": ["control", "control", "treatment", "treatment"],
                "day": [0, 2, 0, 2],
                "exposed": [1, 0, 3, 0],
                "revenue": [99.0, 4.0, 999.0, 6.0],
            }
        ),
        unit="unit",
        group="arm",
        date="day",
        exposure_date="exposed",
        control="control",
        metrics={"revenue": "mean"},
    )
    metric = _metric(source, "revenue")
    totals = {row["group_id"]: row for row in source.moments(metric)}
    assert {(arm, row["n"], _sum_y(row)) for arm, row in totals.items()} == {
        ("control", 2, 4.0),
        ("treatment", 1, 6.0),
    }
    units = nw.from_native(source.unit_frame(metric), eager_only=True).iter_rows(named=True)
    assert {row["unit_id"]: row["y"] for row in units} == {"c0": 0.0, "c1": 4.0, "t1": 6.0}


def test_sparse_compliance_snapshot_keeps_enrolled_units_with_only_future_rows() -> None:
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked", window_days=3),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment affects outcomes only through uptake"
        ),
    )
    source = from_unit_panel(
        pa.table(
            {
                "unit": ["c0", "c1", "t0", "t1"],
                "arm": ["control", "control", "treatment", "treatment"],
                "day": [2, 1, 2, 1],
                "exposed": [0, 0, 0, 0],
                "revenue": [1.0, 2.0, 3.0, 4.0],
                "clicked": [1.0, 1.0, 0.0, 1.0],
            }
        ),
        unit="unit",
        group="arm",
        date="day",
        exposure_date="exposed",
        control="control",
        metrics={"revenue": "mean"},
        uptake="clicked",
        design=design,
    )
    provisional = source.compliance_summary(design=design, as_of=1)
    assert {(arm.group_id, arm.n_units, arm.uptake_total) for arm in provisional.arms} == {
        ("control", 2, 1.0),
        ("treatment", 2, 1.0),
    }
    completed = source.compliance_summary(design=design, as_of=3, completed_windows_only=True)
    assert {(arm.group_id, arm.n_units, arm.uptake_total) for arm in completed.arms} == {
        ("control", 2, 2.0),
        ("treatment", 2, 1.0),
    }


@pytest.mark.parametrize("backend", ["polars", "arrow"])
@pytest.mark.parametrize("value_column", ["revenue", "__day_idx__", "__exposure__"])
def test_panel_quantile_ignores_pre_exposure_decoys(backend: str, value_column: str) -> None:
    rows: list[dict[str, Any]] = []
    for arm in ("control", "treatment"):
        for index in range(40):
            exposure = 1 + index % 2
            identity = {
                "user_id": f"{arm}{index}",
                "variant": arm,
                "exposed_on": exposure,
            }
            rows.append(
                {
                    **identity,
                    "day": exposure - 1,
                    "revenue": 1000.0 if arm == "treatment" else 100.0,
                    "orders": 100.0,
                }
            )
            rows.append(
                {
                    **identity,
                    "day": exposure,
                    "revenue": None
                    if index == 0
                    else float(10 + index % 9 + 2 * (arm == "treatment")),
                    "orders": float(1 + index % 3),
                }
            )
        rows.append(
            {
                "user_id": f"{arm}_future",
                "variant": arm,
                "exposed_on": 3,
                "day": 0,
                "revenue": 10000.0,
                "orders": 100.0,
            }
        )
    admitted = [row for row in rows if row["day"] >= row["exposed_on"]]
    expected_values = [
        0.0 if row["revenue"] is None else row["revenue"]
        for row in sorted(admitted, key=lambda row: row["user_id"])
    ]
    specs = [
        MetricSpec(name="mean", value_column=value_column, missing="zero"),
        MetricSpec(
            name="median", value_column=value_column, type="quantile", quantile=0.5, missing="zero"
        ),
        MetricSpec(
            name="ratio", type="ratio", numerator=value_column, denominator="orders", missing="zero"
        ),
    ]

    def source(data: list[dict[str, Any]]) -> FramePanelSource:
        frame: Any = pa.Table.from_pylist(data)
        if backend == "polars":
            import polars as pl

            frame = pl.DataFrame(data)
        frame = nw.from_native(frame, eager_only=True).rename({"revenue": value_column}).to_native()
        return from_unit_panel(
            frame,
            unit="user_id",
            group="variant",
            date="day",
            exposure_date="exposed_on",
            control="control",
            metrics=specs,
        )

    actual, expected = source(rows), source(admitted)
    for metric in actual.context.metrics:
        observed = nw.from_native(actual.unit_frame(metric), eager_only=True).sort("unit_id")
        oracle = nw.from_native(expected.unit_frame(metric), eager_only=True).sort("unit_id")
        assert observed.to_dict(as_series=False) == oracle.to_dict(as_series=False)
        assert observed["y"].to_list() == expected_values
        assert observed.filter(nw.col("unit_id") == "control0")["y"].item() == 0.0
    oracle_rows = {row.metric: row.require_lift() for row in readouts.run(expected)}
    for row in readouts.run(actual):
        lift, oracle_lift = row.require_lift(), oracle_rows[row.metric]
        assert (lift.value, lift.lb, lift.ub) == pytest.approx(
            (oracle_lift.value, oracle_lift.lb, oracle_lift.ub)
        )


def test_panel_breakout_identity_survives_sparse_zero_fill() -> None:
    frame = pa.table(
        {
            "user_id": ["u1", "u1", "u2"],
            "variant": ["control", "control", "treatment"],
            "day": ["d1", "d2", "d1"],
            "country": ["US", "US", "CA"],
            "plan": ["free", "free", "paid"],
            "revenue": [1.0, 2.0, 3.0],
        }
    )
    src = from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=["country", "plan"],
    )
    assert src.densified_cells == 1
    metric = _metric(src, "revenue")
    day2_by_country = {
        (row["country"], row["group_id"]): row
        for row in src.moments(metric, grain="daily", by=["country"])
        if row["ds"] == "d2"
    }
    assert set(day2_by_country) == {("US", "control"), ("CA", "treatment")}
    # u2's day-2 cell exists only because densification zero-filled it, and it
    # carries u2's own breakout identity rather than a null segment.
    assert day2_by_country[("CA", "treatment")]["n"] == 1
    assert _sum_y(day2_by_country[("CA", "treatment")]) == 0.0
    day2_by_plan = {
        (row["plan"], row["group_id"]): row
        for row in src.moments(metric, grain="daily", by=["plan"])
        if row["ds"] == "d2"
    }
    assert set(day2_by_plan) == {("free", "control"), ("paid", "treatment")}
    assert day2_by_plan[("paid", "treatment")]["n"] == 1
    assert _sum_y(day2_by_plan[("paid", "treatment")]) == 0.0


def test_daily_returns_one_row_per_day_group_metric(panel_frame: pa.Table) -> None:
    analysis = from_unit_panel(
        panel_frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[
            MetricSpec(name="revenue", type="mean"),
            MetricSpec(name="aov", type="ratio", numerator="revenue", denominator="orders"),
        ],
    )
    rows = readouts.daily(analysis)
    keys = [(r["ds"], r["group_id"], r["metric"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert len(keys) == 3 * 2 * 2  # 3 days x 2 groups x 2 metrics


def test_frame_panel_source_moments_are_scoped_to_the_requested_breakout() -> None:
    """moments() memoizes per breakout dimension; a second dimension must not
    be served the first dimension's cached rows."""
    frame = _panel_with_breakouts().append_column("orders", pa.array([1.0] * 8))
    metrics = [
        MetricSpec(name="revenue", type="mean"),
        MetricSpec(name="aov", type="ratio", numerator="revenue", denominator="orders"),
    ]
    src = from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=metrics,
        breakouts=["country", "plan"],
    )
    by_plan_reference = from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=metrics,
        breakouts=["plan"],
    )
    metric = _metric(src, "revenue")

    by_country = {
        (row["country"], row["group_id"]): (row["n"], _sum_y(row))
        for row in src.moments(metric, by=["country"])
    }
    by_plan = {
        (row["plan"], row["group_id"]): (row["n"], _sum_y(row))
        for row in src.moments(metric, by=["plan"])
    }
    expected_plan = {
        (row["plan"], row["group_id"]): (row["n"], _sum_y(row))
        for row in by_plan_reference.moments(_metric(by_plan_reference, "revenue"), by=["plan"])
    }

    assert set(by_country) == {
        ("US", "control"),
        ("US", "treatment"),
        ("CA", "control"),
        ("CA", "treatment"),
    }
    assert by_plan == expected_plan
    # Re-reading the first dimension after the second must not be served the
    # second's rows either.
    assert {
        (row["country"], row["group_id"]): (row["n"], _sum_y(row))
        for row in src.moments(metric, by=["country"])
    } == by_country


def test_frame_panel_source_asof_moments_are_scoped_to_completed_windows_only() -> None:
    """A completed-windows asof read must not be served the provisional rows
    cached by an earlier open-window read of the same source (or vice versa)."""
    exposure = {
        "c1": date(2026, 1, 1),
        "c2": date(2026, 1, 3),
        "t1": date(2026, 1, 1),
        "t2": date(2026, 1, 3),
    }
    days = [date(2026, 1, d) for d in (1, 2, 3, 4)]
    rows = [
        (unit, "control" if unit.startswith("c") else "treatment", day, exposure[unit], 1.0)
        for unit in exposure
        for day in days
        if day >= exposure[unit]
    ]
    src = from_unit_panel(
        pa.table(
            {
                "user_id": [r[0] for r in rows],
                "variant": [r[1] for r in rows],
                "ds": [r[2] for r in rows],
                "exposed_on": [r[3] for r in rows],
                "revenue": [r[4] for r in rows],
            }
        ),
        unit="user_id",
        group="variant",
        date="ds",
        control="control",
        exposure_date="exposed_on",
        metrics=[MetricSpec(name="revenue", window_days=3)],
    )
    metric = _metric(src, "revenue")

    provisional = src.moments(metric, grain="asof")
    decision = src.moments(metric, grain="asof", completed_windows_only=True)

    assert {row["ds"] for row in provisional} == set(days)
    assert {row["ds"] for row in decision} == {date(2026, 1, 4)}
    assert sorted(row["n"] for row in decision) == [1, 1]
    assert {row["ds"] for row in src.moments(metric, grain="asof")} == set(days)


def test_frame_panel_source_moments_returns_defensive_copies(panel_frame: pa.Table) -> None:
    """moments() is memoized per grain (test above); a caller mutating a
    returned row dict must not corrupt the cache for later callers of the
    same or a different metric at that grain."""
    src = from_unit_panel(
        panel_frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", type="mean")],
    )
    metric = src.context.metrics[0]
    first = src.moments(metric, grain="total")
    first[0]["ref_y"] = -999.0
    second = src.moments(metric, grain="total")
    assert second[0]["ref_y"] != -999.0
    asof_first = src.moments(metric, grain="asof")
    asof_first[0]["ref_y"] = -999.0
    asof_second = src.moments(metric, grain="asof")
    assert asof_second[0]["ref_y"] != -999.0


def test_whole_panel_run_equals_summary_run_on_hand_collapsed_data(panel_frame: pa.Table) -> None:
    """run() = collapse to per-unit totals, then the summary moment path - verbatim."""
    panel_result = readouts.run(
        from_unit_panel(
            panel_frame,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
            design=_DESIGN,
        )
    )

    totals: dict[str, float] = {}
    for r in _PANEL_ROWS:
        totals[r[0]] = totals.get(r[0], 0.0) + r[3]
    by_unit = {r[0]: r[1] for r in _PANEL_ROWS}
    collapsed = pa.table(
        {
            "user_id": list(totals),
            "variant": [by_unit[u] for u in totals],
            "revenue": [totals[u] for u in totals],
        }
    )
    summary_result = readouts.run(
        from_unit_summary(
            collapsed,
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
            design=_DESIGN,
        )
    )

    assert len(panel_result) == len(summary_result) == 1
    assert panel_result[0].require_lift().value == pytest.approx(
        summary_result[0].require_lift().value, rel=1e-12
    )
    assert panel_result[0].require_lift().lb == pytest.approx(
        summary_result[0].require_lift().lb, rel=1e-12
    )
    assert panel_result[0].require_lift().ub == pytest.approx(
        summary_result[0].require_lift().ub, rel=1e-12
    )


def test_srm_on_panel_flags_imbalance_and_clears_balance(panel_frame: pa.Table) -> None:
    balanced = readouts.srm(
        from_unit_panel(
            panel_frame,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
            design=_DESIGN,
        ),
    )
    assert isinstance(balanced, SRMResult)
    assert balanced.is_srm is False
    assert balanced.inference == "always_valid"

    # Observed is the fixture's real 4/4 split; declaring a 90/10 EXPECTED
    # allocation against it is a deliberate mismatch regardless of sample size.
    lopsided = readouts.srm(
        from_unit_panel(
            panel_frame,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
            design=_DESIGN,
        ),
        expected={"control": 0.9, "treatment": 0.1},
        inference="fixed",
    )
    assert isinstance(lopsided, SRMResult)
    assert lopsided.is_srm is True


def test_srm_uses_design_declared_allocation_as_default_expected(
    panel_frame: pa.Table,
) -> None:
    """A design that declares its allocation owns the SRM null: with
    expected=None the test runs against the DECLARED shares, not the
    equal-split fallback - so a healthy-looking 50/50 observed split
    against a declared 90/10 design is flagged, and an explicit
    expected= override still wins."""
    design_with_allocation = Randomized(
        control_group="control", allocation={"control": 0.9, "treatment": 0.1}
    )
    src = from_unit_panel(
        panel_frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        design=design_with_allocation,
    )
    # The fixture's real split is 4/4: healthy vs equal split, a mismatch
    # vs the declared 90/10.
    flagged = readouts.srm(src, inference="fixed")
    assert isinstance(flagged, SRMResult)
    assert flagged.expected["control"] == pytest.approx(0.9)
    assert flagged.is_srm is True
    # Explicit expected= still overrides the declaration.
    overridden = readouts.srm(src, expected={"control": 0.5, "treatment": 0.5})
    assert isinstance(overridden, SRMResult)
    assert overridden.expected == {"control": 0.5, "treatment": 0.5}
    assert overridden.is_srm is False


def test_panel_duplicate_unit_day_raises() -> None:
    dup_rows = [*_PANEL_ROWS, _PANEL_ROWS[0]]
    with pytest.raises(InvalidRequestError) as exc:
        from_unit_panel(
            _panel_table(dup_rows),
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
        )
    assert exc.value.code == "source.panel.duplicate_unit_days"


def test_panel_conflicting_group_per_unit_raises(panel_frame: pa.Table) -> None:
    conflicting_rows = [*_PANEL_ROWS[:-1], ("u8", "control", "d3", 8.0, 3.0)]
    with pytest.raises(InvalidRequestError) as exc:
        from_unit_panel(
            _panel_table(conflicting_rows),
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
        )
    assert exc.value.code == "source.panel.multi_group_unit"


def _panel_backend(backend: str, columns: Mapping[str, Sequence[object]]) -> Any:
    table = pa.table(columns)
    if backend == "arrow":
        return table
    if backend == "polars":
        import polars as pl

        return pl.from_arrow(table)
    return table.to_pandas()


@pytest.mark.parametrize("budget", [0, 64 * 1024 * 1024])
def test_daily_accepts_free_form_day_labels_for_every_kernel(
    budget: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import increment.frame as frame_module

    monkeypatch.setattr(frame_module, "_DAY_PANEL_SCRATCH_BUDGET_BYTES", budget)
    source = from_unit_panel(
        pa.table(
            {
                "unit": ["u1", "u2"],
                "arm": ["control", "control"],
                "day": ["2025-9", "2025-10"],
                "value": [1.0, 2.0],
            }
        ),
        unit="unit",
        group="arm",
        date="day",
        control="control",
        metrics={"value": "mean"},
    )
    rows = source.moments(_metric(source, "value"), grain="daily")
    assert {row["ds"] for row in rows} == {"2025-9", "2025-10"}


@pytest.mark.parametrize("budget", [0, 64 * 1024 * 1024])
def test_panel_unit_identity_does_not_sort_mixed_public_ids(
    budget: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pandas as pd

    import increment.frame as frame_module

    monkeypatch.setattr(frame_module, "_DAY_PANEL_SCRATCH_BUDGET_BYTES", budget)
    frame = pd.DataFrame(
        {
            "unit": pd.Series([1, "two"], dtype=object),
            "arm": ["control", "control"],
            "day": [0, 1],
            "value": [1.0, 2.0],
        }
    )
    source = from_unit_panel(
        frame,
        unit="unit",
        group="arm",
        date="day",
        control="control",
        metrics={"value": "mean"},
    )
    rows = source.moments(_metric(source, "value"), grain="daily")
    assert {row["ds"] for row in rows} == {0, 1}
    assert all(row["n"] == 2 for row in rows)

    asof_rows = source.moments(_metric(source, "value"), grain="asof")
    assert {row["ds"] for row in asof_rows} == {0, 1}
    assert all(row["n"] == 2 for row in asof_rows)


@pytest.mark.parametrize("budget", [0, 64 * 1024 * 1024])
def test_completed_ratio_asof_mixed_unit_ids_use_dense_and_streamed_kernels(
    budget: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import narwhals as nw
    import pandas as pd

    import increment.frame as frame_module

    monkeypatch.setattr(frame_module, "_DAY_PANEL_SCRATCH_BUDGET_BYTES", budget)
    units = [1, "two", 2, "three"]
    groups = ["control", "treatment", "control", "treatment"]
    frame = pd.DataFrame(
        {
            "unit": pd.Series([unit for unit in units for _ in range(3)], dtype=object),
            "arm": [group for group in groups for _ in range(3)],
            "day": [day for _ in units for day in range(3)],
            "exposed": [0] * 12,
            "numerator": [
                float(unit_index + 1) * (day + 1) for unit_index in range(4) for day in range(3)
            ],
            "denominator": [float(unit_index + 1) for unit_index in range(4) for _ in range(3)],
        }
    )
    source = from_unit_panel(
        frame,
        unit="unit",
        group="arm",
        date="day",
        exposure_date="exposed",
        control="control",
        metrics=[
            MetricSpec(
                name="ratio",
                type="ratio",
                numerator="numerator",
                denominator="denominator",
                window_days=2,
            )
        ],
    )
    original_join = nw.DataFrame.join
    join_count = 0

    def reorder_join(self, other, *args, **kwargs):
        nonlocal join_count
        joined = original_join(self, other, *args, **kwargs)
        if "unit_id" in self.columns and "unit_id" in other.columns:
            ordinal = next(
                (name for name in joined.columns if name.startswith("__unit_ordinal")),
                None,
            )
            if ordinal is not None:
                join_count += 1
                current = joined.get_column(ordinal).to_list()
                day = self.get_column("ds").to_list()[0] if "ds" in self.columns else join_count
                offset = (int(day) + 1) % len(current)
                rotated = current[offset:] + current[:offset]
                ranks = {value: index for index, value in enumerate(rotated)}
                helper = f"__join_order_{join_count}"
                return joined.with_columns(
                    nw.new_series(
                        helper, [ranks[value] for value in current], backend=joined.implementation
                    )
                ).sort(helper)
        return joined

    monkeypatch.setattr(nw.DataFrame, "join", reorder_join)
    rows = source.moments(_metric(source, "ratio"), grain="asof", completed_windows_only=True)
    assert join_count > 0
    assert {row["ds"] for row in rows} == {2}
    assert {row["group_id"] for row in rows} == {"control", "treatment"}
    assert all(row["n"] == 2 for row in rows)
    by_group = {row["group_id"]: row for row in rows}
    assert _sum_y(by_group["control"]) == pytest.approx(12.0)
    assert _sum_y(by_group["treatment"]) == pytest.approx(18.0)
    assert _sum_den(by_group["control"]) == pytest.approx(8.0)
    assert _sum_den(by_group["treatment"]) == pytest.approx(12.0)


@pytest.mark.parametrize("backend", ["arrow", "polars", "pandas"])
@pytest.mark.parametrize("exposure_mode", ["explicit", "inferred"])
def test_streamed_asof_restores_identity_after_reordered_exposure_joins(
    backend: str, exposure_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import narwhals as nw

    import increment.frame as frame_module
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    days = range(6)
    sparse_rows = []
    dense_rows = []
    for index in range(6):
        exposure = index % 3
        for day in range(exposure, 6):
            event = float(index % 2 == 0 and day == exposure + 1)
            row = {
                "unit": f"u{index}",
                "arm": "control" if index < 3 else "treatment",
                "day": day,
                "outcome": event,
                "clicked": event,
            }
            if exposure_mode == "explicit":
                row["exposed"] = exposure
            dense_rows.append(row)
            if day in (exposure, exposure + 1, 4, 5):
                sparse_rows.append(row)

    specs = [
        MetricSpec(
            name="outcome",
            type="mean",
            value_column="outcome",
            window_days=3 if exposure_mode == "explicit" else None,
        )
    ]
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked", window_days=3),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment affects outcome only through uptake"
        ),
    )
    exposure_column = "exposed" if exposure_mode == "explicit" else None

    def build(rows: list[dict[str, Any]]) -> FramePanelSource:
        columns = {name: [row[name] for row in rows] for name in rows[0]}
        return from_unit_panel(
            _panel_backend(backend, columns),
            unit="unit",
            group="arm",
            date="day",
            exposure_date=exposure_column,
            control="control",
            metrics=specs,
            uptake="clicked",
            design=design,
            observation_end=max(days),
        )

    def read(source: FramePanelSource) -> dict[tuple[Any, ...], dict[str, Any]]:
        metric = _metric(source, "outcome")
        return {
            (completed, row["ds"], row["group_id"]): row
            for completed in ((False, True) if exposure_mode == "explicit" else (False,))
            for row in source.moments(metric, grain="asof", completed_windows_only=completed)
        }

    expected = read(build(dense_rows))
    sparse_source = build(sparse_rows)
    monkeypatch.setattr(frame_module, "_DAY_PANEL_SCRATCH_BUDGET_BYTES", 0)
    original_join = nw.DataFrame.join
    join_count = 0

    def reorder_join(self, other, *args, **kwargs):
        nonlocal join_count
        joined = original_join(self, other, *args, **kwargs)
        if "unit_id" in self.columns and "unit_id" in other.columns:
            join_count += 1
            ordinal = next(
                (name for name in joined.columns if name.startswith("__unit_ordinal")),
                None,
            )
            if ordinal is not None:
                current = joined.get_column(ordinal).to_list()
                offset = join_count % len(current)
                rotated = current[offset:] + current[:offset]
                ranks = {value: index for index, value in enumerate(rotated)}
                helper = f"__join_order_{join_count}"
                return joined.with_columns(
                    nw.new_series(
                        helper, [ranks[value] for value in current], backend=joined.implementation
                    )
                ).sort(helper)
        return joined

    monkeypatch.setattr(nw.DataFrame, "join", reorder_join)
    actual = read(sparse_source)
    assert join_count > 0
    _assert_sparse_panel_moments_match(actual, expected, f"{backend}/{exposure_mode}")
    assert any(row["sum_d"] > 0.0 for row in actual.values())


@pytest.mark.parametrize("backend", ["arrow", "polars", "pandas"])
def test_streamed_asof_allows_outcome_and_uptake_to_share_a_column(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import increment.frame as frame_module
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    monkeypatch.setattr(frame_module, "_DAY_PANEL_SCRATCH_BUDGET_BYTES", 0)
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked", window_days=2),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment affects outcomes only through uptake"
        ),
    )
    source = from_unit_panel(
        _panel_backend(
            backend,
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["control", "control", "treatment", "treatment"],
                "day": [0, 0, 1, 1],
                "exposed": [0, 0, 1, 1],
                "clicked": [1.0, 0.0, 0.0, 1.0],
            },
        ),
        unit="unit",
        group="arm",
        date="day",
        exposure_date="exposed",
        control="control",
        metrics=[
            MetricSpec(
                name="clicked_conversion", type="conversion", value_column="clicked", window_days=2
            )
        ],
        uptake="clicked",
        design=design,
    )
    metric = _metric(source, "clicked_conversion")
    rows = source.moments(metric, grain="asof")
    assert rows and any(row["sum_d"] > 0.0 for row in rows)


@pytest.mark.parametrize("backend", ["arrow", "polars", "pandas"])
@pytest.mark.parametrize("canonical_name", ["unit_id", "group_id", "ds"])
def test_panel_refuses_metric_declarations_colliding_with_canonical_roles(
    backend: str, canonical_name: str
) -> None:
    """A value column must not collide with a projected canonical role."""
    import copy
    import pickle

    columns = {
        "user_id": ["u1", "u1", "u2", "u2"],
        "variant": ["control", "control", "treatment", "treatment"],
        "day": ["d1", "d2", "d1", "d2"],
        "unit_id": [0.0, 1.0, 0.0, 1.0],
        "group_id": [1.0, 2.0, 3.0, 4.0],
        "ds": [5.0, 6.0, 7.0, 8.0],
        "revenue": [1.0, 2.0, 3.0, 4.0],
    }
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_panel(
            _panel_backend(backend, columns),
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[MetricSpec(name="revenue", value_column=canonical_name)],
        )
    for error in (
        exc_info.value,
        copy.deepcopy(exc_info.value),
        pickle.loads(pickle.dumps(exc_info.value)),
    ):
        assert error.code == "frame.panel.declaration_canonical_collision"
        assert error.context["role"] == "metric value"
        assert error.context["canonical_name"] == canonical_name
        assert error.context["source_name"] == canonical_name
        with pytest.raises(TypeError):
            cast(Any, error.context)["role"] = "changed"


@pytest.mark.parametrize("backend", ["arrow", "polars", "pandas"])
@pytest.mark.parametrize("canonical_name", ["unit_id", "group_id", "ds"])
def test_panel_refuses_uptake_declarations_colliding_with_canonical_roles(
    backend: str, canonical_name: str
) -> None:
    columns = {
        "user_id": ["u1", "u1", "u2", "u2"],
        "variant": ["control", "control", "treatment", "treatment"],
        "day": ["d1", "d2", "d1", "d2"],
        "unit_id": [0.0, 1.0, 0.0, 1.0],
        "group_id": [0.0, 1.0, 0.0, 1.0],
        "ds": [0.0, 1.0, 0.0, 1.0],
        "revenue": [1.0, 2.0, 3.0, 4.0],
    }
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_panel(
            _panel_backend(backend, columns),
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            uptake=canonical_name,
            metrics=[MetricSpec(name="revenue")],
        )
    assert exc_info.value.code == "frame.panel.declaration_canonical_collision"
    assert exc_info.value.context["role"] == "uptake"
    assert exc_info.value.context["canonical_name"] == canonical_name


@pytest.mark.parametrize("backend", ["arrow", "polars", "pandas"])
@pytest.mark.parametrize("canonical_name", ["unit_id", "group_id", "ds"])
def test_panel_refuses_ratio_denominator_declarations_colliding_with_canonical_roles(
    backend: str, canonical_name: str
) -> None:
    columns = {
        "user_id": ["u1", "u1", "u2", "u2"],
        "variant": ["control", "control", "treatment", "treatment"],
        "day": ["d1", "d2", "d1", "d2"],
        "revenue": [1.0, 2.0, 3.0, 4.0],
        "unit_id": [0.0, 1.0, 0.0, 1.0],
        "group_id": [1.0, 2.0, 3.0, 4.0],
        "ds": [5.0, 6.0, 7.0, 8.0],
    }
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_panel(
            _panel_backend(backend, columns),
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[
                MetricSpec(
                    name="ratio",
                    type="ratio",
                    numerator="revenue",
                    denominator=canonical_name,
                )
            ],
        )
    assert exc_info.value.code == "frame.panel.declaration_canonical_collision"
    assert exc_info.value.context["role"] == "metric denominator"
    assert exc_info.value.context["canonical_name"] == canonical_name


@pytest.mark.parametrize("backend", ["arrow", "polars", "pandas"])
def test_panel_allows_roles_named_as_their_canonical_columns(backend: str) -> None:
    source = from_unit_panel(
        _panel_backend(
            backend,
            {
                "unit_id": ["u1", "u1", "u2", "u2"],
                "group_id": ["control", "control", "treatment", "treatment"],
                "ds": ["d1", "d2", "d1", "d2"],
                "revenue": [1.0, 2.0, 3.0, 4.0],
            },
        ),
        unit="unit_id",
        group="group_id",
        date="ds",
        control="control",
        metrics=[MetricSpec(name="revenue")],
    )
    rows = source.moments(_metric(source, "revenue"))
    assert {row["group_id"]: _sum_y(row) for row in rows} == pytest.approx(
        {"control": 3.0, "treatment": 7.0}
    )


@pytest.mark.parametrize("backend", ["arrow", "polars", "pandas"])
@pytest.mark.parametrize("covariate", ["ds", "group_id"])
def test_panel_allows_cuped_covariates_named_as_canonical_columns(
    backend: str, covariate: str
) -> None:
    from increment.estimation.engine import Method, estimate_lift

    columns = {
        "user_id": [f"u{i}" for i in range(40) for _ in range(2)],
        "variant": ["control" if i % 2 == 0 else "treatment" for i in range(40) for _ in range(2)],
        "day": [day for _ in range(40) for day in ("d1", "d2")],
        "revenue": [10.0 + i % 7 + (i * i) % 5 for i in range(40) for _ in range(2)],
    }
    covariates = [float(i % 7) for i in range(40) for _ in range(2)]
    moments = []
    estimates = []
    for name in ("baseline", covariate):
        source = from_unit_panel(
            _panel_backend(backend, {**columns, name: covariates}),
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[MetricSpec(name="revenue", covariate=name)],
            design=_DESIGN,
        )
        metric = _metric(source, "revenue")
        rows = source.moments(metric, include_covariate=True)
        moments.append({row["group_id"]: row for row in rows})
        result = estimate_lift(
            metrics=[metric],
            summary=rows,
            control_group="control",
            methods=[Method(name="cuped", variance_reduction="cuped")],
        )
        assert not result.failures
        (estimate,) = result.results
        estimates.append(estimate)
    for group_id, row in moments[1].items():
        _assert_moments_agree(
            row, moments[0][group_id], ("n", "ref_y", "cy1", "cy2", "ref_x", "cx1", "cx2", "cxy")
        )
    ordinary, canonical = estimates
    for result in estimates:
        assert result.abs_lb is not None and result.abs_ub is not None
        assert result.require_lift().lb is not None and result.require_lift().ub is not None
    assert canonical.abs_reference_kind == ordinary.abs_reference_kind
    assert canonical.reference_kind == ordinary.reference_kind
    assert (canonical.abs_diff, canonical.abs_lb, canonical.abs_ub) == pytest.approx(
        (ordinary.abs_diff, ordinary.abs_lb, ordinary.abs_ub), rel=1e-9, abs=1e-12
    )
    actual, expected = canonical.require_lift(), ordinary.require_lift()
    assert (actual.value, actual.lb, actual.ub) == pytest.approx(
        (expected.value, expected.lb, expected.ub), rel=1e-9, abs=1e-12
    )


def test_panel_missing_columns_are_all_reported_at_once(panel_frame: pa.Table) -> None:
    with pytest.raises(InvalidRequestError) as exc:
        from_unit_panel(
            panel_frame,
            unit="nope_unit",
            group="variant",
            date="nope_date",
            control="control",
            metrics={"revenue": "mean"},
        )
    assert exc.value.code == "source.frame.metric_missing"
    assert _reported_missing(exc.value) == {"nope_unit", "nope_date"}


@pytest.mark.parametrize("bad_value", [0.5, -1.0])
def test_panel_conversion_metric_must_be_binary(bad_value: float) -> None:
    table = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["control", "control", "treatment", "treatment"],
            "day": ["d1"] * 4,
            "converted": [0.0, bad_value, 1.0, 0.0],
        }
    )
    with pytest.raises(InvalidRequestError) as exc:
        from_unit_panel(
            table,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[MetricSpec(name="converted", type="conversion")],
        )
    assert exc.value.code == "source.frame.conversion_not_binary"


def test_panel_conversion_accepts_bool_and_declared_zero_nulls() -> None:
    table = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["control", "control", "treatment", "treatment"],
            "day": ["d1"] * 4,
            "converted": [True, False, True, None],
        }
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="converted", type="conversion", missing="zero")],
    )
    out = {r["group_id"]: r for r in src.moments(_metric(src, "converted"), grain="total")}
    assert out["control"]["n"] == 2
    assert out["treatment"]["n"] == 2
    assert _sum_y(out["control"]) == pytest.approx(1.0)
    assert _sum_y(out["treatment"]) == pytest.approx(1.0)


def test_panel_conversion_total_grain_refuses_multi_day_collapse() -> None:
    """Summing per-day conversion cells collapses a multi-day converter to
    a day count, not a 0/1 flag; refused instead of silently misreporting."""
    table = pa.table(
        {
            "user_id": ["u1", "u1", "u2"],
            "variant": ["control", "control", "treatment"],
            "day": ["d1", "d2", "d1"],
            "converted": [1.0, 1.0, 0.0],
        }
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="converted", type="conversion")],
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        src.moments(_metric(src, "converted"), grain="total")
    assert exc_info.value.code == "frame.moments.metric_type_conversion"
    assert exc_info.value.context["metric"] == "converted"
    assert exc_info.value.context["value"] == pytest.approx(2.0)


def test_panel_invalid_conversion_sibling_is_deferred_until_selected() -> None:
    table = pa.table(
        {
            "user_id": ["c0", "c0", "c1", "c1", "t0", "t0", "t1", "t1"],
            "variant": ["control"] * 4 + ["treatment"] * 4,
            "day": [0, 1] * 4,
            "revenue": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
            "converted": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    revenue_only = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", type="mean")],
    )
    with_sibling = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[
            MetricSpec(name="revenue", type="mean"),
            MetricSpec(name="converted", type="conversion"),
        ],
    )
    revenue = _metric(revenue_only, "revenue")
    revenue_with_sibling = _metric(with_sibling, "revenue")
    assert with_sibling.moments(revenue_with_sibling, grain="total") == revenue_only.moments(
        revenue, grain="total"
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        with_sibling.moments(_metric(with_sibling, "converted"), grain="total")
    assert exc_info.value.code == "frame.moments.metric_type_conversion"
    assert exc_info.value.context["metric"] == "converted"
    assert exc_info.value.context["value"] == pytest.approx(2.0)


def test_panel_absent_control_lists_observed_groups(panel_frame: pa.Table) -> None:
    with pytest.raises(InvalidRequestError) as exc:
        from_unit_panel(
            panel_frame,
            unit="user_id",
            group="variant",
            date="day",
            control="holdout",
            metrics={"revenue": "mean"},
        )
    assert exc.value.code == "source.frame.control_missing"


def test_panel_unknown_metric_uses_totals_refusal_code(panel_frame: pa.Table) -> None:
    src = from_unit_panel(
        panel_frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )

    class _Fake:
        name = "not_declared"

    with pytest.raises(InvalidRequestError) as exc_info:
        src.unit_frame(_Fake())  # ty: ignore[invalid-argument-type]
    assert exc_info.value.code == "frame.frame_totals.metric_was_declared"


def test_panel_covariate_on_windowed_metric_raises_and_points_at_summary(
    panel_frame: pa.Table,
) -> None:
    """A windowed metric has no per-unit collapse to attach a covariate
    to -- refused by name, distinct from the plain-metric path which now
    resolves the covariate."""
    rows = [
        ("u1", "control", "2025-01-01", "2025-01-01", 4.0, 2.0),
        ("u2", "treatment", "2025-01-01", "2025-01-01", 6.0, 2.0),
    ]
    frame = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "exposed_on", "revenue", "orders"],
                zip(*rows, strict=True),
                strict=True,
            )
        )
    )
    with pytest.raises(InvalidRequestError) as exc:
        from_unit_panel(
            frame,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[MetricSpec(name="revenue", covariate="orders", window_days=1)],
            exposure_date="exposed_on",
        )
    assert exc.value.code == "frame.validation.from_unit_panel"
    assert exc.value.context["covered"] == ("revenue",)


def test_panel_covariate_varying_within_unit_refuses(panel_frame: pa.Table) -> None:
    """`orders` varies across u1's own rows (2.0, 3.0, 0.0) -- a plain
    metric's covariate resolves per-unit like `unit_frame()`'s does, and
    refuses the same way when it is not actually constant."""
    src = from_unit_panel(
        panel_frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", covariate="orders")],
    )
    with pytest.raises(InvalidRequestError) as exc:
        src.moments(_metric(src, "revenue"), grain="total")
    assert exc.value.code == "frame.frame_panel.unit_covariate_varies"


def test_panel_covariate_varies_with_mixed_public_unit_ids() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "unit": pd.Series([1, 1, "two", "two"], dtype=object),
            "arm": ["control", "control", "treatment", "treatment"],
            "day": [0, 1, 0, 1],
            "revenue": [1.0, 2.0, 3.0, 4.0],
            "orders": [1.0, 2.0, 3.0, 4.0],
        }
    )
    source = from_unit_panel(
        frame,
        unit="unit",
        group="arm",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", covariate="orders")],
    )

    with pytest.raises(InvalidRequestError) as exc:
        source.moments(_metric(source, "revenue"), grain="total")

    assert exc.value.code == "frame.frame_panel.unit_covariate_varies"
    assert exc.value.context["units"] == ("two", 1)


def test_panel_cuped_moments_and_interval_match_unit_summary() -> None:
    """A per-unit-constant covariate wires CUPED through the panel's
    total-grain collapse: identical centered moments, and the same
    cuped_adjust() interval, as the same per-unit data through
    from_unit_summary."""
    import numpy as np

    rng = np.random.default_rng(11)
    n = 30
    units = [f"u{i}" for i in range(n)]
    tenure = {u: float(rng.normal(100, 15)) for u in units}
    group = {u: ("control" if i % 2 == 0 else "treatment") for i, u in enumerate(units)}
    revenue = {u: 5.0 + 0.05 * tenure[u] + float(rng.normal(0, 1)) for u in units}

    panel_rows = []
    for u in units:
        for day in ("d1", "d2"):
            panel_rows.append(
                {
                    "user_id": u,
                    "variant": group[u],
                    "day": day,
                    "revenue": revenue[u] / 2.0,
                    "tenure": tenure[u],
                }
            )
    panel_table = pa.Table.from_pylist(panel_rows)
    summary_table = pa.Table.from_pylist(
        [
            {"user_id": u, "variant": group[u], "revenue": revenue[u], "tenure": tenure[u]}
            for u in units
        ]
    )
    spec = MetricSpec(name="revenue", covariate="tenure")

    panel_src = from_unit_panel(
        panel_table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[spec],
        design=_DESIGN,
    )
    summary_src = from_unit_summary(
        summary_table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[spec],
        design=_DESIGN,
    )
    panel_moments = {
        r["group_id"]: r for r in panel_src.moments(_metric(panel_src, "revenue"), grain="total")
    }
    summary_moments = {
        r["group_id"]: r
        for r in summary_src.moments(_metric(summary_src, "revenue"), grain="total")
    }
    assert set(panel_moments) == set(summary_moments) == {"control", "treatment"}
    for group_id, got in panel_moments.items():
        _assert_moments_agree(
            got,
            summary_moments[group_id],
            ("n", "ref_y", "cy1", "cy2", "ref_x", "cx1", "cx2", "cxy"),
        )

    from increment.estimation.engine import Method, estimate_lift

    cuped_methods = [Method(name="cuped", variance_reduction="cuped")]
    panel_result = estimate_lift(
        metrics=list(panel_src.context.metrics),
        summary=list(panel_moments.values()),
        control_group="control",
        methods=cuped_methods,
    ).results
    summary_result = estimate_lift(
        metrics=list(summary_src.context.metrics),
        summary=list(summary_moments.values()),
        control_group="control",
        methods=cuped_methods,
    ).results
    assert len(panel_result) == len(summary_result) == 1
    panel_lift = panel_result[0].require_lift()
    summary_lift = summary_result[0].require_lift()
    assert panel_lift.value == pytest.approx(summary_lift.value, rel=1e-9)
    assert panel_lift.lb == pytest.approx(summary_lift.lb, rel=1e-9)
    assert panel_lift.ub == pytest.approx(summary_lift.ub, rel=1e-9)


def test_panel_covariate_varying_within_unit_does_not_block_sibling_metrics(
    panel_frame: pa.Table,
) -> None:
    """`orders` varies across u1's own rows, but a sibling metric that
    declares no covariate must still read: the covariate-varies refusal is
    deferred per metric, the same way a collapsed-percentile-bound
    refusal is, instead of blocking every total-grain read on the source."""
    src = from_unit_panel(
        panel_frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[
            MetricSpec(name="revenue", covariate="orders"),
            MetricSpec(name="clicks", value_column="revenue"),
        ],
    )
    rows = src.moments(_metric(src, "clicks"), grain="total")
    assert {r["group_id"] for r in rows} == {"control", "treatment"}
    with pytest.raises(InvalidRequestError) as exc:
        src.moments(_metric(src, "revenue"), grain="total")
    assert exc.value.code == "frame.frame_panel.unit_covariate_varies"


def test_panel_cuped_covariate_name_colliding_with_a_metric_column() -> None:
    """A covariate whose name matches a co-registered metric's own value
    column must read the resolved per-unit covariate, not that metric's
    panel-summed total."""
    import numpy as np

    rng = np.random.default_rng(7)
    n = 40
    units = [f"u{i}" for i in range(n)]
    tenure: dict[str, float] = {u: float(rng.normal(100, 15)) for u in units}
    group = {u: ("control" if i % 2 == 0 else "treatment") for i, u in enumerate(units)}
    revenue = {u: 5.0 + 0.05 * tenure[u] + float(rng.normal(0, 1)) for u in units}

    panel_rows = [
        {
            "user_id": u,
            "variant": group[u],
            "day": day,
            "revenue": revenue[u] / 2.0,
            "tenure": tenure[u],
        }
        for u in units
        for day in ("d1", "d2")
    ]
    panel_table = pa.Table.from_pylist(panel_rows)
    summary_table = pa.Table.from_pylist(
        [
            {"user_id": u, "variant": group[u], "revenue": revenue[u], "tenure": tenure[u]}
            for u in units
        ]
    )
    metrics = [
        MetricSpec(name="tenure", missing="zero"),
        MetricSpec(name="revenue", covariate="tenure"),
    ]
    panel_src = from_unit_panel(
        panel_table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=metrics,
        design=_DESIGN,
    )
    summary_src = from_unit_summary(
        summary_table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=metrics,
        design=_DESIGN,
    )
    panel_moments = {
        r["group_id"]: r for r in panel_src.moments(_metric(panel_src, "revenue"), grain="total")
    }
    summary_moments = {
        r["group_id"]: r
        for r in summary_src.moments(_metric(summary_src, "revenue"), grain="total")
    }
    for group_id, got in panel_moments.items():
        _assert_moments_agree(
            got,
            summary_moments[group_id],
            ("n", "ref_y", "cy1", "cy2", "ref_x", "cx1", "cx2", "cxy"),
        )


@pytest.mark.parametrize("backend", ["arrow", "polars", "pandas"])
def test_panel_cuped_scratch_alias_collision_matches_ordinary_and_summary(backend: str) -> None:
    """Scratch-like metrics cannot replace either of two declared covariates."""
    import numpy as np

    from increment.estimation.engine import Method, estimate_lift

    rng = np.random.default_rng(17)
    outcomes = {"revenue": "tenure", "spend": "activity"}
    unit_rows = []
    for i in range(40):
        tenure, activity = rng.normal([100, 20], [15, 4])
        unit_rows.append(
            {
                "user_id": f"u{i}",
                "variant": "control" if i % 2 == 0 else "treatment",
                "tenure": float(tenure),
                "activity": float(activity),
                "revenue": float(5 + 0.05 * tenure + rng.normal()),
                "spend": float(10 + 0.2 * activity + rng.normal()),
            }
        )

    def native(data: list[dict[str, Any]]) -> Any:
        table = pa.Table.from_pylist(data)
        if backend == "polars":
            import polars as pl

            return pl.from_arrow(table)
        return table.to_pandas() if backend == "pandas" else table

    def sources(prefix: str) -> tuple[FramePanelSource, FrameTotalsSource]:
        decoys = [f"{prefix}{name}" for name in outcomes.values()]
        metrics = [
            MetricSpec(name=name, covariate=covariate) for name, covariate in outcomes.items()
        ]
        metrics += [MetricSpec(name=name, missing="zero") for name in decoys]
        summary_rows = [
            {**row, **dict.fromkeys(decoys, float(i % 3))} for i, row in enumerate(unit_rows)
        ]
        panel_rows = [
            {**row, "day": day, **{name: float(row[name]) / 2 for name in (*outcomes, *decoys)}}
            for row in summary_rows
            for day in ("d1", "d2")
        ]
        return (
            from_unit_panel(
                native(panel_rows),
                unit="user_id",
                group="variant",
                date="day",
                control="control",
                metrics=metrics,
                design=_DESIGN,
            ),
            from_unit_summary(
                native(summary_rows),
                unit="user_id",
                group="variant",
                control="control",
                metrics=metrics,
                design=_DESIGN,
            ),
        )

    ordinary_panel, ordinary_summary = sources("scratch_")
    adversarial_panel, adversarial_summary = sources("__covariate__")
    all_sources = (ordinary_summary, ordinary_panel, adversarial_panel, adversarial_summary)
    columns = ("n", "ref_y", "cy1", "cy2", "ref_x", "cx1", "cx2", "cxy")
    for metric_name in outcomes:
        moments = [
            {
                row["group_id"]: row
                for row in source.moments(
                    _metric(source, metric_name), grain="total", include_covariate=True
                )
            }
            for source in all_sources
        ]
        for rows in moments:
            assert set(rows) == {"control", "treatment"}
            for group_id, row in rows.items():
                _assert_moments_agree(row, moments[0][group_id], columns)
        results = []
        for source, rows in zip(all_sources, moments, strict=True):
            computation = estimate_lift(
                metrics=[_metric(source, metric_name)],
                summary=list(rows.values()),
                control_group="control",
                methods=[Method(name="cuped", variance_reduction="cuped")],
            )
            assert not computation.failures
            (result,) = computation.results
            results.append(result)
        baseline = results[0]
        for result in results[1:]:
            assert result.reference_kind == baseline.reference_kind
            assert result.relative_unavailable_reason == baseline.relative_unavailable_reason
            assert result.abs_reference_kind == baseline.abs_reference_kind
            assert result.note == baseline.note
            for field in (
                "reference_df",
                "abs_reference_df",
                "abs_diff",
                "abs_se",
                "abs_lb",
                "abs_ub",
                "abs_alpha",
            ):
                got, want = getattr(result, field), getattr(baseline, field)
                assert got == (
                    pytest.approx(want, rel=1e-9, abs=1e-12) if want is not None else None
                )
            got, want = result.require_lift(), baseline.require_lift()
            assert got.value == pytest.approx(want.value, rel=1e-9, abs=1e-12)
            assert got.lb == pytest.approx(want.lb, rel=1e-9, abs=1e-12)
            assert got.ub == pytest.approx(want.ub, rel=1e-9, abs=1e-12)


def test_panel_covariate_all_nan_for_a_unit_matches_summary_mean_imputation() -> None:
    """A unit whose covariate is NaN on every one of its rows is constant
    and missing, not varying: NaN must be normalized to null before the
    constancy check, the same way `_apply_spec_missing` normalizes it
    before mean-imputation. A pyarrow panel used to refuse this as
    `unit_covariate_varies` since pyarrow's NaN != NaN and its `count()`
    treats NaN as non-null."""
    import math

    import numpy as np

    rng = np.random.default_rng(3)
    n = 20
    units = [f"u{i}" for i in range(n)]
    tenure: dict[str, float] = {u: float(rng.normal(50, 10)) for u in units}
    tenure["u3"] = math.nan
    group = {u: ("control" if i % 2 == 0 else "treatment") for i, u in enumerate(units)}
    revenue = {u: 5.0 + float(rng.normal(0, 1)) for u in units}

    panel_rows = [
        {
            "user_id": u,
            "variant": group[u],
            "day": day,
            "revenue": revenue[u] / 2.0,
            "tenure": tenure[u],
        }
        for u in units
        for day in ("d1", "d2")
    ]
    panel_table = pa.Table.from_pylist(panel_rows)
    summary_table = pa.Table.from_pylist(
        [
            {"user_id": u, "variant": group[u], "revenue": revenue[u], "tenure": tenure[u]}
            for u in units
        ]
    )
    spec = MetricSpec(name="revenue", covariate="tenure")

    with pytest.warns(IncrementWarning) as rec_panel:
        panel_src = from_unit_panel(
            panel_table,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[spec],
            design=_DESIGN,
        )
    assert "frame.validation.metric_covariate_missing_imputed" in warning_codes(rec_panel)
    with pytest.warns(IncrementWarning) as rec_summary:
        summary_src = from_unit_summary(
            summary_table,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[spec],
            design=_DESIGN,
        )
    assert "frame.validation.metric_covariate_missing_imputed" in warning_codes(rec_summary)
    panel_moments = {
        r["group_id"]: r for r in panel_src.moments(_metric(panel_src, "revenue"), grain="total")
    }
    summary_moments = {
        r["group_id"]: r
        for r in summary_src.moments(_metric(summary_src, "revenue"), grain="total")
    }
    for group_id, got in panel_moments.items():
        _assert_moments_agree(
            got,
            summary_moments[group_id],
            ("n", "ref_y", "cy1", "cy2", "ref_x", "cx1", "cx2", "cxy"),
        )


def test_panel_retention_type_is_rejected() -> None:
    """The terse retention form refuses on the panel path too - it cannot
    carry the required threshold_days, so the failure fires in
    ``coerce_metrics`` before ``MetricSpec`` (and this module's own
    windowing/retention machinery) is ever reached.
    """
    with pytest.raises(InvalidRequestError) as exc:
        from_unit_panel(
            _panel_table(),
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "retention"},
        )
    assert exc.value.code == "frame.metric_type_retention"
    assert exc.value.context["name"] == "revenue"


def test_panel_retention_computes_real_semantics_given_exposure_date() -> None:
    """A validly-constructed retention MetricSpec computes real
    per-unit retention on the panel path given an explicit
    exposure_date - distinct from test_panel_retention_type_is_rejected
    above, whose terse-form spec never reaches this guard because
    ``coerce_metrics`` rejects the underspecified terse form first (it
    cannot carry ``threshold_days``).
    """
    base = date(2025, 1, 1)
    # u1 returns on day 7 (inside the unbounded band [7, inf)); u2 never
    # returns after day 0; u3 (control) never returns after day 0 either.
    rows = [
        ("u1", "treatment", base, base, 1.0),
        ("u1", "treatment", base + timedelta(days=7), base, 1.0),
        ("u2", "treatment", base, base, 1.0),
        ("u3", "control", base, base, 1.0),
    ]
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(zip(["user_id", "variant", "day", "exposed_on", "revenue"], cols, strict=True))
    )
    spec = MetricSpec(name="d7", type="retention", value_column="revenue", threshold_days=7)
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[spec],
        exposure_date="exposed_on",
    )
    metric = _metric(src, "d7")
    by_group = {r["group_id"]: r for r in src.moments(metric, grain="total")}
    assert by_group["treatment"]["n"] == 2
    assert _sum_y(by_group["treatment"]) == pytest.approx(1.0)  # only u1 returned on/after day 7
    assert by_group["control"]["n"] == 1
    assert _sum_y(by_group["control"]) == pytest.approx(0.0)


def test_asof_completed_bounded_retention_before_band_close_returns_no_rows() -> None:
    """Final-only bounded retention has no snapshot before close on any backend."""
    base = date(2025, 1, 1)
    rows = [
        ("c1", "control", base, base, 0.0),
        ("c1", "control", base + timedelta(days=6), base, 0.0),
        ("t1", "treatment", base, base, 0.0),
        ("t1", "treatment", base + timedelta(days=6), base, 0.0),
    ]
    table = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "exposed_on", "returned"],
                zip(*rows, strict=True),
                strict=True,
            )
        )
    )
    frames: list[tuple[str, Any]] = [("pyarrow", table)]
    if importlib.util.find_spec("pandas") is not None:
        frames.append(("pandas", table.to_pandas()))
    if importlib.util.find_spec("polars") is not None:
        import polars as pl

        frames.append(("polars", pl.from_arrow(table)))

    for backend, frame in frames:
        src = from_unit_panel(
            frame,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[
                MetricSpec(
                    name="d7_to_d14",
                    type="retention",
                    value_column="returned",
                    threshold_days=(7, 14),
                )
            ],
            exposure_date="exposed_on",
        )

        assert (
            src.moments(
                _metric(src, "d7_to_d14"),
                grain="asof",
                completed_windows_only=True,
            )
            == []
        ), backend


def test_asof_completed_bounded_retention_with_uptake_requires_uptake_window() -> None:
    """The no-uptake completion path must not swallow the pre-existing
    uptake-finalization guard: a bounded retention metric with an uptake
    fact declared still needs a bounded uptake window to finalize sum_d,
    exactly like a windowed mean does."""
    base = date(2025, 1, 1)
    rows = [
        ("c1", "control", base, base, 0.0, 0.0),
        ("c1", "control", base + timedelta(days=6), base, 1.0, 1.0),
        ("t1", "treatment", base, base, 0.0, 0.0),
        ("t1", "treatment", base + timedelta(days=6), base, 1.0, 1.0),
    ]
    table = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "exposed_on", "returned", "clicked"],
                zip(*rows, strict=True),
                strict=True,
            )
        )
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[
            MetricSpec(
                name="d0_to_d2", type="retention", value_column="returned", threshold_days=(0, 2)
            )
        ],
        exposure_date="exposed_on",
        uptake="clicked",
        design=_DESIGN,
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        src.moments(_metric(src, "d0_to_d2"), grain="asof", completed_windows_only=True)
    assert exc_info.value.code == "frame.moments.asof_completed_requires_uptake_window"


def test_asof_completed_retention_skips_co_registered_unbounded_metric() -> None:
    """Completed bounded/mean reads survive an unbounded sibling's refusal."""
    base = date(2025, 1, 1)
    rows = [
        ("c1", "control", base, base, 0.0),
        ("c1", "control", base + timedelta(days=1), base, 1.0),
        ("c1", "control", base + timedelta(days=2), base, 0.0),
        ("t1", "treatment", base, base, 0.0),
        ("t1", "treatment", base + timedelta(days=1), base, 0.0),
        ("t1", "treatment", base + timedelta(days=2), base, 0.0),
    ]
    table = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "exposed_on", "revenue"],
                zip(*rows, strict=True),
                strict=True,
            )
        )
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[
            MetricSpec(name="revenue", type="mean"),
            MetricSpec(
                name="d0_to_d2",
                type="retention",
                value_column="revenue",
                threshold_days=(0, 2),
            ),
            MetricSpec(
                name="d0_plus",
                type="retention",
                value_column="revenue",
                threshold_days=0,
            ),
        ],
        exposure_date="exposed_on",
    )
    mean = _metric(src, "revenue")
    bounded = _metric(src, "d0_to_d2")
    unbounded = _metric(src, "d0_plus")

    assert src.moments(mean, grain="asof", completed_windows_only=True)
    assert src.moments(bounded, grain="asof", completed_windows_only=True)
    with pytest.raises(InvalidRequestError) as exc:
        src.moments(unbounded, grain="asof", completed_windows_only=True)
    assert exc.value.code == "frame.frame_panel.asof_moments_completed"


def test_windowed_metric_requires_exposure_date(panel_frame: pa.Table) -> None:
    with pytest.raises(InvalidRequestError) as exc:
        from_unit_panel(
            panel_frame,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[MetricSpec(name="revenue", window_days=7)],
        )
    assert exc.value.code == "source.panel.exposure_date"


def test_exposure_date_must_be_constant_per_unit() -> None:
    rows = [
        ("u1", "control", date(2025, 1, 1), date(2025, 1, 1), 4.0),
        ("u1", "control", date(2025, 1, 2), date(2025, 1, 2), 5.0),  # differs from row above
    ]
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(zip(["user_id", "variant", "day", "exposed_on", "revenue"], cols, strict=True))
    )
    with pytest.raises(InvalidRequestError) as exc:
        from_unit_panel(
            table,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
            exposure_date="exposed_on",
        )
    assert exc.value.code == "source.panel.exposure_date"


def test_censoring_warns_past_threshold() -> None:
    base = date(2025, 1, 1)
    rows = []
    for i in range(10):
        offset = i % 5
        fe = base + timedelta(days=offset)
        group = "control" if i % 2 == 0 else "treatment"
        rows.append((f"u{i}", group, fe, fe, 1.0))
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(zip(["user_id", "variant", "day", "exposed_on", "revenue"], cols, strict=True))
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", window_days=3)],
        exposure_date="exposed_on",
        observation_end=base + timedelta(days=3),
    )
    metric = _metric(src, "revenue")
    with pytest.warns(IncrementWarning) as rec:
        src.moments(metric, grain="total")
    assert "frame.censoring.dropped_units" in warning_codes(rec)


def test_retention_daily_grain_refused() -> None:
    base = date(2025, 1, 1)
    rows = [
        ("u1", "control", base, base, 1.0),
        ("u1", "control", base + timedelta(days=8), base, 1.0),
        ("u2", "treatment", base, base, 1.0),
    ]
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(zip(["user_id", "variant", "day", "exposed_on", "revenue"], cols, strict=True))
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="d7", type="retention", value_column="revenue", threshold_days=7)],
        exposure_date="exposed_on",
    )
    metric = _metric(src, "d7")
    with pytest.raises(CapabilityError):
        src.moments(metric, grain="daily")


def test_retention_daily_grain_does_not_block_other_metrics_on_the_same_source() -> None:
    """A co-registered retention metric must not block ``grain="daily"``
    reads for every OTHER metric on the same source.
    ``FramePanelSource.moments()`` used to batch ``_daily_moment_rows``
    across ALL of ``self._specs`` the first time ANY metric asked for
    ``grain="daily"``, raising ``CapabilityError`` if the batch
    contained one retention spec - even when the requested metric
    (``revenue``) never asked about retention. The rejection must be
    scoped to the metric the caller actually requested.
    """
    base = date(2025, 1, 1)
    rows = [
        ("u1", "control", base, base, 1.0),
        ("u1", "control", base + timedelta(days=1), base, 2.0),
        ("u2", "treatment", base, base, 3.0),
    ]
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(zip(["user_id", "variant", "day", "exposed_on", "revenue"], cols, strict=True))
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[
            MetricSpec(name="revenue", type="mean"),
            MetricSpec(name="d7", type="retention", value_column="revenue", threshold_days=7),
        ],
        exposure_date="exposed_on",
    )
    mean_metric = _metric(src, "revenue")
    retention_metric = _metric(src, "d7")

    # The metric actually requested (mean, not retention) reads fine;
    # the co-registered retention spec never blocks it.
    daily_rows = src.moments(mean_metric, grain="daily")
    assert daily_rows
    assert all(r["metric"] == "revenue" for r in daily_rows)
    by_ds = {r["ds"]: r for r in daily_rows if r["group_id"] == "control"}
    assert _sum_y(by_ds[base]) == pytest.approx(1.0)
    assert _sum_y(by_ds[base + timedelta(days=1)]) == pytest.approx(2.0)

    # The retention metric itself still correctly refuses grain="daily",
    # whether asked before or after the mean metric populates the cache.
    with pytest.raises(CapabilityError):
        src.moments(retention_metric, grain="daily")


def test_bounded_retention_denominator_includes_a_mature_unit_with_no_in_band_rows() -> None:
    """A mature, censoring-surviving unit whose only densified rows fall
    entirely OUTSIDE its bounded retention band ``[0, band_end)`` must
    still count in the denominator (as "observed, did not return",
    ``y=0``), not silently vanish.

    Before the fix, the retention branch of ``_reduce_spec`` derived its
    ``units`` denominator universe from the day-0..right-edge-BOUNDED
    frame - the same ``bounded`` frame the non-retention day-0 left-bound
    fix filters to ``[0, band_end)``. A unit whose only in-band rows are
    absent from the panel's globally-observed date set (a real gap, not a
    zero-fill artefact) would have zero rows in ``bounded`` and so drop
    out of ``units`` entirely - undercounting ``n`` - even though it
    passed per-unit censoring and genuinely belongs in the denominator.

    u1 anchors the panel's globally-observed dates at 0,1,2,3,4,8,9,10
    (skipping 5,6,7 - the exact window a later-exposed unit will need).
    u2 is exposed on day 5 with ``threshold_days=(0, 3)``: its band is
    ``[5, 8)``, which the global date set never covers (5,6,7 are
    missing), so EVERY one of u2's densified rows lands outside the band
    (day_idx in {-5,-4,-3,-2,-1,3,4,5}, none in ``[0, 3)``). u2's
    observable end (fact-scoped max observed date, day 10) is still well
    past its maturity bound (day 5 + 3 = day 8), so it survives censoring
    and belongs in the denominator with ``y=0``.
    """
    rows: list[tuple[str, str, date, date, float]] = [
        ("u1", "control", base_u1, date(2025, 1, 1), 1.0)
        for base_u1 in (
            date(2025, 1, 1),
            date(2025, 1, 2),
            date(2025, 1, 3),
            date(2025, 1, 4),
            date(2025, 1, 5),
            date(2025, 1, 9),
            date(2025, 1, 10),
            date(2025, 1, 11),
        )
    ]
    rows.append(("u2", "control", date(2025, 1, 1), date(2025, 1, 6), 1.0))
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(zip(["user_id", "variant", "day", "exposed_on", "revenue"], cols, strict=True))
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[
            MetricSpec(name="d3b", type="retention", value_column="revenue", threshold_days=(0, 3))
        ],
        exposure_date="exposed_on",
    )
    metric = _metric(src, "d3b")
    moments = src.moments(metric, grain="total")
    assert len(moments) == 1
    assert moments[0]["n"] == 2, "u2 must count in the denominator despite having no in-band rows"


def test_bounded_conversion_denominator_includes_a_mature_unit_with_no_in_window_rows() -> None:
    """Same structural gap as the retention test above, transposed to a
    windowed ``type="conversion"`` metric: ``_reduce_spec``'s conversion
    branch also only emits a row for a unit with at least one row inside
    ``bounded`` (``[0, window_days)``), so it shares the identical
    denominator undercount the retention fix addressed. u2's window
    ``[Jan6, Jan9)`` lands exactly on the dates u1's real activity skips
    (Jan6/7/8), so u2 has zero rows inside its own window and, before the
    generalised fix, would silently drop from both numerator and
    denominator despite surviving censoring.
    """
    rows: list[tuple[str, str, date, date, float]] = [
        ("u1", "control", base_u1, date(2025, 1, 1), 1.0)
        for base_u1 in (
            date(2025, 1, 1),
            date(2025, 1, 2),
            date(2025, 1, 3),
            date(2025, 1, 4),
            date(2025, 1, 5),
            date(2025, 1, 9),
            date(2025, 1, 10),
            date(2025, 1, 11),
        )
    ]
    rows.append(("u2", "control", date(2025, 1, 1), date(2025, 1, 6), 1.0))
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(zip(["user_id", "variant", "day", "exposed_on", "revenue"], cols, strict=True))
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="conv", type="conversion", value_column="revenue", window_days=3)],
        exposure_date="exposed_on",
    )
    metric = _metric(src, "conv")
    moments = src.moments(metric, grain="total")
    assert len(moments) == 1
    assert moments[0]["n"] == 2, "u2 must count in the denominator despite having no in-window rows"
    assert _sum_y(moments[0]) == pytest.approx(1.0), "only u1 converted; u2 reads y=0, not absent"


def test_bounded_mean_denominator_includes_a_mature_unit_with_no_in_window_rows() -> None:
    """Same structural gap, transposed to a windowed ``type="mean"``
    metric: the plain ``windowed.group_by(...).agg(*aggs)`` path also
    only emits a row for a unit with at least one row inside ``bounded``.
    """
    rows: list[tuple[str, str, date, date, float]] = [
        ("u1", "control", base_u1, date(2025, 1, 1), 10.0)
        for base_u1 in (
            date(2025, 1, 1),
            date(2025, 1, 2),
            date(2025, 1, 3),
            date(2025, 1, 4),
            date(2025, 1, 5),
            date(2025, 1, 9),
            date(2025, 1, 10),
            date(2025, 1, 11),
        )
    ]
    rows.append(("u2", "control", date(2025, 1, 1), date(2025, 1, 6), 10.0))
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(zip(["user_id", "variant", "day", "exposed_on", "revenue"], cols, strict=True))
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", window_days=3)],
        exposure_date="exposed_on",
    )
    metric = _metric(src, "revenue")
    moments = src.moments(metric, grain="total")
    assert len(moments) == 1
    assert moments[0]["n"] == 2, "u2 must count in the denominator despite having no in-window rows"
    assert _sum_y(moments[0]) == pytest.approx(30.0), "u1's 3 in-window days sum; u2 contributes 0"


def test_windowed_breakout_moments_recover_sparse_censored_zeroes() -> None:
    """Each breakout keeps mature units that lack in-window facts."""
    base = date(2025, 1, 1)
    rows = [
        ("us_active", "control", base, base, "US", 2.0),
        ("us_active", "control", base + timedelta(days=2), base, "US", 4.0),
        # Establish the shared observed endpoint without entering this window.
        ("us_active", "control", base + timedelta(days=10), base, "US", 100.0),
        ("us_empty", "control", base, base + timedelta(days=5), "US", 9.0),
        ("ca_active", "control", base + timedelta(days=1), base, "CA", 7.0),
        ("ca_empty", "control", base, base + timedelta(days=5), "CA", 9.0),
    ]
    cols = list(zip(*rows, strict=True))
    src = from_unit_panel(
        pa.table(
            dict(
                zip(
                    ["user_id", "variant", "day", "exposed_on", "country", "revenue"],
                    cols,
                    strict=True,
                )
            )
        ),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", window_days=3)],
        exposure_date="exposed_on",
        breakouts=["country"],
    )

    moments = {row["country"]: row for row in src.moments(_metric(src, "revenue"), by=["country"])}

    assert set(moments) == {"US", "CA"}
    assert {country: row["n"] for country, row in moments.items()} == {"US": 2, "CA": 2}
    assert _sum_y(moments["US"]) == pytest.approx(6.0)
    assert _sum_y(moments["CA"]) == pytest.approx(7.0)
    assert moments["US"]["ref_y"] == pytest.approx(3.0)
    assert moments["CA"]["ref_y"] == pytest.approx(3.5)


def test_bounded_ratio_denominator_includes_a_mature_unit_with_no_in_window_rows() -> None:
    """Same structural gap as the retention/conversion/mean tests above,
    transposed to a windowed ``type="ratio"`` metric: end-to-end, pinning
    the correct group-level numbers (``n``, the recovered sum(y) and
    sum(den)) once u2 is correctly re-added to the denominator.

    Does NOT, on its own, discriminate whether ``__y_den__`` specifically
    is filled to ``0.0`` vs left null for the re-added unit: narwhals'
    ``sum()`` skips nulls identically to an explicit ``0.0`` fill on
    every backend this project supports (verified directly against
    ``increment/query/builders.py``'s parity-tested reference and against
    narwhals itself - ``nw.col(...).sum()`` over a column containing one
    null among real values, or over an all-null column, returns the same
    value a ``0.0``-filled column would), so this aggregate-level
    assertion is unaffected by that specific choice. See
    ``test_reduce_spec_ratio_fills_missing_denominator_to_zero_not_null``
    below for the test that actually discriminates it, at the
    ``_reduce_spec`` level where the null-vs-zero distinction is still
    directly observable, before aggregation masks it.
    """
    rows: list[tuple[str, str, date, date, float, float]] = [
        ("u1", "control", base_u1, date(2025, 1, 1), 10.0, 5.0)
        for base_u1 in (
            date(2025, 1, 1),
            date(2025, 1, 2),
            date(2025, 1, 3),
            date(2025, 1, 4),
            date(2025, 1, 5),
            date(2025, 1, 9),
            date(2025, 1, 10),
            date(2025, 1, 11),
        )
    ]
    rows.append(("u2", "control", date(2025, 1, 1), date(2025, 1, 6), 10.0, 5.0))
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "exposed_on", "revenue", "cost"],
                cols,
                strict=True,
            )
        )
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[
            MetricSpec(
                name="rev_ratio",
                type="ratio",
                numerator="revenue",
                denominator="cost",
                window_days=3,
            )
        ],
        exposure_date="exposed_on",
    )
    metric = _metric(src, "rev_ratio")
    moments = src.moments(metric, grain="total")
    assert len(moments) == 1
    assert moments[0]["n"] == 2, "u2 must count in the denominator despite having no in-window rows"
    assert _sum_y(moments[0]) == pytest.approx(30.0), "u1's 3 in-window days sum; u2 contributes 0"
    assert _sum_den(moments[0]) == pytest.approx(15.0), (
        "group denominator pinned to u1's contribution"
    )


def test_reduce_spec_ratio_fills_missing_denominator_to_zero_not_null() -> None:
    """Direct unit test of ``_reduce_spec``'s ratio branch, at the level
    where the ``fill_zero`` choice is still observable: the per-unit
    output row, before ``_windowed_moment_rows`` hands it to
    ``_additive_moments`` for aggregation (where ``sum()``'s null-skipping
    would mask a dropped ``__y_den__`` fill - see the aggregate-level
    sibling test's docstring above). u2 has no row in ``windowed`` at all
    (simulating zero in-window rows) but IS present in ``units_source``
    (simulating a censoring survivor); the fix must fill BOTH ``__y__``
    and ``__y_den__`` to ``0.0`` for it, not just ``__y__`` - a
    regression that dropped ``__y_den__`` from ``fill_zero`` would leave
    it ``None`` here, silently passing every existing end-to-end test.
    """
    from increment._frame_moments import _reduce_spec

    spec = MetricSpec(name="rev_ratio", type="ratio", numerator="revenue", denominator="cost")
    windowed = nw.from_native(
        pa.table({"unit_id": ["u1"], "group_id": ["control"], "revenue": [10.0], "cost": [5.0]}),
        eager_only=True,
    )
    units_source = nw.from_native(
        pa.table({"unit_id": ["u1", "u2"], "group_id": ["control", "control"]}),
        eager_only=True,
    )
    out = _reduce_spec(windowed, spec, band=None, units_source=units_source)
    by_unit = {row["unit_id"]: row for row in out.iter_rows(named=True)}
    assert by_unit["u1"]["__y__"] == pytest.approx(10.0)
    assert by_unit["u1"]["__y_den__"] == pytest.approx(5.0)
    assert by_unit["u2"]["__y__"] == pytest.approx(0.0)
    assert by_unit["u2"]["__y_den__"] == pytest.approx(0.0), (
        "u2 has no in-window row -- __y_den__ must fill to 0.0, not stay null"
    )


def test_daily_grain_windowed_metric_bounds_only_at_the_right_edge() -> None:
    """Important Finding 1 (final whole-branch review): no automated test
    exercised grain="daily" together with window_days on an ordinary
    (non-retention) metric - despite _daily_moment_rows' documented
    right-edge-only contract (day_idx < window_days; never whole-unit
    censoring, never a retention band's left edge). Reproduces the
    reviewer's manual check: a 6-day panel for one unit with
    window_days=3 - days 0/1/2 (inside the half-open window) must be
    present with the unit's real per-day value, days 3/4/5 (at/past the
    right edge) must be entirely absent from the daily output, not
    zeroed and not merely excluded from a sum.
    """
    base = date(2025, 1, 1)
    rows = [("u1", "control", base + timedelta(days=d), base, float(d) + 1.0) for d in range(6)]
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(zip(["user_id", "variant", "day", "exposed_on", "revenue"], cols, strict=True))
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", window_days=3)],
        exposure_date="exposed_on",
    )
    metric = _metric(src, "revenue")
    daily_rows = src.moments(metric, grain="daily")
    by_ds = {r["ds"]: r for r in daily_rows}

    in_window = [base + timedelta(days=d) for d in range(3)]
    past_window = [base + timedelta(days=d) for d in range(3, 6)]

    assert set(by_ds) == set(in_window), (
        "daily rows must cover exactly the in-window days, no more, no less"
    )
    for d, ds in enumerate(in_window):
        assert by_ds[ds]["n"] == 1
        assert _sum_y(by_ds[ds]) == pytest.approx(float(d) + 1.0)
    for ds in past_window:
        assert ds not in by_ds, f"day {ds} is at/past the window's right edge -- must be absent"


def test_daily_grain_windowed_metric_excludes_pre_exposure_days() -> None:
    """Daily-grain half: the daily
    windowed filter bounded only the right edge (``day_idx <
    window_days``), never day 0's left edge, so a unit's pre-exposure
    panel rows (``day_idx < 0``) manufactured spurious calendar-day
    buckets in the daily output - inflating per-day ``n`` even for a
    zero-fill row, since daily ``n`` counts rows per ``(ds, group)`` and
    the ibis reference's spine has no such days at all (it starts at
    day 0 by construction). u1 has 3 PRE-exposure days (base..base+2,
    day_idx -3..-1) before its own exposure on base+3, then 3 in-window
    days (day_idx 0..2) under window_days=3.
    """
    base = date(2025, 1, 1)
    exposed_on = base + timedelta(days=3)
    rows = [
        ("u1", "control", base + timedelta(days=d), exposed_on, float(d) + 1.0) for d in range(6)
    ]
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(zip(["user_id", "variant", "day", "exposed_on", "revenue"], cols, strict=True))
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", window_days=3)],
        exposure_date="exposed_on",
    )
    metric = _metric(src, "revenue")
    daily_rows = src.moments(metric, grain="daily")
    by_ds = {r["ds"]: r for r in daily_rows}

    pre_exposure_days = [base + timedelta(days=d) for d in range(3)]
    in_window = [exposed_on + timedelta(days=d) for d in range(3)]

    for ds in pre_exposure_days:
        assert ds not in by_ds, (
            f"day {ds} is PRE-exposure (before exposed_on={exposed_on}) -- must not "
            "appear as a spurious daily bucket"
        )
    assert set(by_ds) == set(in_window), (
        "daily rows must cover exactly the in-window days, no more, no less"
    )
    for d, ds in enumerate(in_window):
        assert by_ds[ds]["n"] == 1
        assert _sum_y(by_ds[ds]) == pytest.approx(float(d + 3) + 1.0)


def test_windowed_censoring_fallback_is_scoped_to_that_metrics_own_fact_stream() -> None:
    """The running-experiment
    censoring fallback (``observation_end=None``) must be resolved from
    EACH windowed spec's OWN fact stream, never from the whole densified
    panel. ``from_unit_panel``'s panel is a WIDE unit-day frame - one
    row per (unit, day), one column per metric, densified with a
    uniform ``fill_null(0.0)`` - so ``panel["ds"].max()`` is identical
    for every metric regardless of which metric's column actually had a
    row there.

    'm' is genuinely observed every day through day 20; 'rare' is
    genuinely observed only on day 0 and day 2 (``None`` afterwards,
    not a real zero - so density never invents a false positive here).
    Both are ``window_days=10``, every unit exposed on day 0, no
    ``observation_end``. Before the fix, ``panel["ds"].max()`` == day 20
    (borrowed from 'm''s own dense rows), so 'rare' read as fully mature
    off dates it never itself reached: every unit admitted, a real (if
    truncated) sum, and NO censoring warning. After the fix, 'rare's own
    fact-scoped observable end is day 2 - short of its own 10-day
    maturity bound - so every enrolled unit is censored and a
    ``UserWarning`` fires.
    """
    base = date(2025, 1, 1)
    rows = []
    for i in range(10):
        unit = f"u{i}"
        group = "control" if i % 2 == 0 else "treatment"
        for d in range(21):  # day 0..20 - 'm' is observed every one of these
            rows.append(
                {
                    "user_id": unit,
                    "variant": group,
                    "day": base + timedelta(days=d),
                    "exposed_on": base,
                    "m": 1.0,
                    "rare": 1.0 if d in (0, 2) else None,
                }
            )
    table = pa.Table.from_pylist(rows)

    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[
            MetricSpec(name="m", window_days=10),
            # 'rare' has explicit nulls on days with no event; the entry
            # chokepoint refuses undeclared nulls so declaring missing="zero" normalizes values only - which days count as observed is still read off the raw non-null rows (the fact-scoped censoring fallback under test).
            MetricSpec(name="rare", window_days=10, missing="zero"),
        ],
        exposure_date="exposed_on",
    )
    m_metric = _metric(src, "m")
    rare_metric = _metric(src, "rare")

    # 'rare''s own fact stream stops at day 2, short of its 10-day maturity
    # bound, so every enrolled unit is censored and a UserWarning fires naming the running-experiment cause; before the fix panel["ds"].max() borrowed 'm''s day-20 rows and 'rare' silently read every unit as mature.
    with pytest.warns(IncrementWarning) as rec:
        rare_rows = src.moments(rare_metric, grain="total")
    assert "frame.censoring.dropped_units" in warning_codes(rec)
    assert sum(r["n"] for r in rare_rows) == 0

    # 'm''s own fact stream reaches day 20, well past its 10-day maturity
    # bound, so it is untouched by 'rare''s censoring even in the same batched pass (FramePanelSource.moments' per-grain memoization).
    m_rows = src.moments(m_metric, grain="total")
    assert sum(r["n"] for r in m_rows) == 10


class TestDayAxisNonStringLabels:
    """A label this cannot justify an order for must refuse, not fall through
    to identity (lexicographic) order."""

    def test_byte_string_labels_refuse_rather_than_sorting_lexicographically(self) -> None:
        # b"01/01/2026" sorts before b"12/31/2025", the same corruption the
        # string path refuses.
        frame = pa.table(
            {
                "user_id": ["u1", "u2"],
                "variant": ["control", "treatment"],
                "day": pa.array([b"12/31/2025", b"01/01/2026"], type=pa.binary()),
                "revenue": [1.0, 2.0],
            }
        )
        src = from_unit_panel(
            frame,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
        )
        with pytest.raises(CapabilityError) as raised:
            src.moments(_metric(src, "revenue"), grain="asof")
        assert raised.value.code == "frame.asof.day_axis_unorderable"

    def test_date_labels_are_still_ordered_by_identity(self) -> None:
        """Calendar labels need no refusal and order chronologically, not by
        the frame's own row order."""
        rows = [
            ("u1", "control", date(2026, 1, 1), 1.0),
            ("u1", "control", date(2025, 12, 31), 2.0),
        ]
        frame = pa.table(
            dict(
                zip(["user_id", "variant", "day", "revenue"], zip(*rows, strict=True), strict=True)
            )
        )
        src = from_unit_panel(
            frame,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
        )
        moments = src.moments(_metric(src, "revenue"), grain="asof")

        assert [row["ds"] for row in moments] == [date(2025, 12, 31), date(2026, 1, 1)]
        assert [_sum_y(row) for row in moments] == [pytest.approx(2.0), pytest.approx(3.0)]


class TestNonFiniteExposureAnchor:
    """The exposure anchor is the origin every day index is measured from, so
    an infinite value there corrupts the same arithmetic as an infinite day."""

    def _frame(self, exposure: float) -> pa.Table:
        return pa.table(
            {
                "user_id": ["u1", "u1", "u2", "u2"],
                "variant": ["control", "control", "treatment", "treatment"],
                "revenue": [1.0, 2.0, 3.0, 4.0],
                "exposure_date": [exposure, exposure, 0.0, 0.0],
                "day": [0.0, 1.0, 0.0, 1.0],
            }
        )

    def _call(self, exposure: float) -> dict[str, int]:
        return from_unit_panel(
            self._frame(exposure),
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
            exposure_date="exposure_date",
        ).unit_counts()

    def test_infinite_exposure_anchor_refuses_at_construction(self) -> None:
        # Previously accepted: the unit kept counting while its day indexes were
        # all infinite, silently excluding its outcomes.
        with pytest.raises(InvalidRequestError) as exc:
            self._call(float("inf"))
        assert exc.value.code == "source.frame.non_finite"

    def test_a_finite_anchor_is_unaffected(self) -> None:
        assert self._call(0.0) == {"control": 1, "treatment": 1}


class TestDayAxisUnhashableLabels:
    """bytearray and a writable memoryview are unhashable, so the byte-like
    check must run before deduping or the caller gets a raw hashing error."""

    def test_bytearray_labels_refuse_rather_than_failing_to_hash(self) -> None:
        with pytest.raises(CapabilityError) as raised:
            _day_axis_label_order([bytearray(b"12/31/2025"), bytearray(b"01/01/2026")])
        assert raised.value.code == "frame.asof.day_axis_unorderable"

    def test_writable_memoryview_labels_refuse(self) -> None:
        with pytest.raises(CapabilityError) as raised:
            _day_axis_label_order([memoryview(bytearray(b"a")), memoryview(bytearray(b"b"))])
        assert raised.value.code == "frame.asof.day_axis_unorderable"


def test_asof_completed_windows_only_gates_a_windowed_mean() -> None:
    """completed_windows_only=True admits a unit only once its own outcome
    window has closed, uptake or no uptake. c1/t1 (exposed Jan 1, 3-day
    window) finalize on Jan 4; c2/t2 (exposed Jan 3) never do, so the decision
    series is one date with one unit per arm."""
    units = ["c1", "c2", "t1", "t2"]
    exposure = {
        "c1": date(2026, 1, 1),
        "c2": date(2026, 1, 3),
        "t1": date(2026, 1, 1),
        "t2": date(2026, 1, 3),
    }
    days = [date(2026, 1, d) for d in (1, 2, 3, 4)]
    rows = [
        (unit, "control" if unit.startswith("c") else "treatment", day, exposure[unit], 1.0)
        for unit in units
        for day in days
        if day >= exposure[unit]
    ]
    table = pa.table(
        {
            "user_id": [r[0] for r in rows],
            "variant": [r[1] for r in rows],
            "ds": [r[2] for r in rows],
            "exposed_on": [r[3] for r in rows],
            "revenue": [r[4] for r in rows],
        }
    )
    source = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="ds",
        control="control",
        exposure_date="exposed_on",
        metrics=[MetricSpec(name="revenue", window_days=3)],
    )
    metric = source.context.metrics[0]

    provisional = source.moments(metric, grain="asof")
    decision = source.moments(metric, grain="asof", completed_windows_only=True)

    assert {row["ds"] for row in provisional} == set(days)
    assert {row["ds"] for row in decision} == {date(2026, 1, 4)}
    assert sorted(row["n"] for row in decision) == [1, 1]


def test_frame_panel_source_snapshots_metrics_list_at_construction(panel_frame: pa.Table) -> None:
    """The first reduction must use construction-time metadata, not a warmed cache."""
    base = from_unit_panel(
        panel_frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )
    owned_metrics = [MetricSpec(name="revenue")]
    owned_counts = base.unit_counts()
    source = FramePanelSource(
        panel=cast(Any, base._sparse_panel),
        metrics=owned_metrics,
        control="control",
        experiment_id=base.context.study_id,
        design=base.design,
        plan=base.plan,
        group_counts=owned_counts,
        n_filled=base.densified_cells,
        synthesised_metrics=base.context.metrics,
    )
    metric = next(m for m in base.context.metrics if m.name == "revenue")
    before = base.moments(metric, grain="total")
    owned_metrics.append(MetricSpec(name="orders"))
    owned_metrics.clear()
    owned_counts.clear()
    after = source.moments(metric, grain="total")
    assert after == before
    assert source.unit_counts() == base.unit_counts()
    source.unit_counts().clear()
    assert source.unit_counts() == base.unit_counts()


def test_frame_panel_source_snapshots_fact_dates_before_first_reduction() -> None:
    start = date(2025, 1, 1)
    frame = pa.Table.from_pylist(
        [
            {
                "unit": unit,
                "arm": "control" if unit < 2 else "treatment",
                "day": start + timedelta(days=day),
                "exposed_on": start,
                "revenue": float(unit + day + 1),
            }
            for unit in range(4)
            for day in range(3)
        ]
    )
    spec = MetricSpec(name="revenue", window_days=2)
    base = from_unit_panel(
        frame,
        unit="unit",
        group="arm",
        date="day",
        control="control",
        metrics=[spec],
        exposure_date="exposed_on",
    )
    owned_dates = dict(base._fact_max_ds)
    source = FramePanelSource(
        panel=base._sparse_panel,
        metrics=[spec],
        control="control",
        experiment_id=base.context.study_id,
        design=base.design,
        plan=base.plan,
        group_counts=base.unit_counts(),
        n_filled=base.densified_cells,
        first_exposure=base._first_exposure,
        exposure=base._exposure,
        fact_max_ds=owned_dates,
        synthesised_metrics=base.context.metrics,
    )
    owned_dates["revenue"] = start
    rows = source.moments(_metric(source, "revenue"), grain="total")
    assert sum(row["n"] for row in rows) == 4


@pytest.mark.parametrize(
    "aliases",
    [
        ("__day_idx__", "__exposure__"),
        ("__observable_days__", "__exposure___right"),
        ("___observable_days__", "__observable_days__"),
    ],
    ids=["day-anchor", "maturity-numerator", "maturity-denominator"],
)
@pytest.mark.parametrize("backend", ["polars", "arrow"])
def test_panel_alias_scratch_columns_match_ordinary_columns_across_grains(
    backend: str, aliases: tuple[str, str]
) -> None:
    base = date(2026, 1, 1)
    rows = [
        {
            "user_id": unit,
            "variant": group,
            "day": base + timedelta(days=offset),
            "exposed_on": base + timedelta(days=exposure),
            "revenue": float(value),
            "orders": float(den),
        }
        for unit, group, exposure, values in (
            ("c", "control", 0, ((0.25, 0.5), (3, 1), (0, 0))),
            ("t", "treatment", 1, ((100, 10), (0.75, 0.25), (0, 0))),
        )
        for offset, (value, den) in enumerate(values)
    ]

    def build(alias: bool) -> FramePanelSource:
        data = [dict(row) for row in rows]
        value, denominator = aliases if alias else ("revenue", "orders")
        if alias:
            for row in data:
                row[value] = row.pop("revenue")
                row[denominator] = row.pop("orders")
        frame: Any = pa.Table.from_pylist(data)
        if backend == "polars":
            import polars as pl

            frame = pl.DataFrame(data)
        metrics = [
            MetricSpec(
                name="mean",
                value_column=value,
                window_days=2,
            ),
            MetricSpec(
                name="ratio",
                type="ratio",
                numerator=value,
                denominator=denominator,
                window_days=2,
            ),
            MetricSpec(name="plain_mean", value_column=value),
            MetricSpec(
                name="plain_ratio",
                type="ratio",
                numerator=value,
                denominator=denominator,
            ),
        ]
        return from_unit_panel(
            frame,
            unit="user_id",
            group="variant",
            date="day",
            exposure_date="exposed_on",
            control="control",
            metrics=metrics,
        )

    ordinary, aliased = build(False), build(True)

    def moment_key(row):
        return str(row.get("ds")), row["group_id"]

    for grain in ("daily", "asof", "total"):
        for name in ("mean", "ratio"):
            left = ordinary.moments(_metric(ordinary, name), grain=grain)
            right = aliased.moments(_metric(aliased, name), grain=grain)
            assert sorted(right, key=moment_key) == sorted(left, key=moment_key)
    for name in ("plain_mean", "plain_ratio"):
        left = nw.from_native(ordinary.unit_frame(_metric(ordinary, name)), eager_only=True).sort(
            "unit_id"
        )
        right = nw.from_native(aliased.unit_frame(_metric(aliased, name)), eager_only=True).sort(
            "unit_id"
        )
        assert right.to_dict(as_series=False) == left.to_dict(as_series=False)
        assert right["y"].to_list() == [3.25, 0.75]
        if name == "plain_ratio":
            assert right["y_den"].to_list() == [1.5, 0.25]


def test_ratio_maturity_requires_both_component_horizons() -> None:
    """A ratio window is complete only through the shared observed horizon."""
    rows = [
        ("c", "control", "d0", "d0", 1.0, 1.0),
        ("c", "control", "d2", "d0", 1.0, None),
        ("t", "treatment", "d0", "d0", 1.0, 1.0),
        ("t", "treatment", "d2", "d0", 0.0, None),
    ]
    table = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "exposed_on", "num", "den"],
                list(zip(*rows, strict=True)),
                strict=True,
            )
        )
    )
    spec = MetricSpec(
        name="ratio",
        type="ratio",
        numerator="num",
        denominator="den",
        window_days=3,
        missing="zero",
    )
    source = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        exposure_date="exposed_on",
        control="control",
        metrics=[spec],
    )
    with pytest.warns(IncrementWarning) as caught:
        assert source.moments(_metric(source, "ratio"), grain="total") == []
    assert "frame.censoring.dropped_units" in warning_codes(caught)

    completed = [(u, g, d, e, n, 1.0) for u, g, d, e, n, _ in rows]
    table = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "exposed_on", "num", "den"],
                list(zip(*completed, strict=True)),
                strict=True,
            )
        )
    )
    source = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        exposure_date="exposed_on",
        control="control",
        metrics=[spec],
    )
    moments = source.moments(_metric(source, "ratio"), grain="total")
    assert {r["group_id"] for r in moments} == {"control", "treatment"}
    assert {r["n"] for r in moments} == {1}
    assert sorted(_sum_y(r) for r in moments) == [1.0, 2.0]
    assert {_sum_den(r) for r in moments} == {2.0}


def test_asof_completed_ratio_uses_shared_component_horizon() -> None:
    """A lagging denominator must not admit a newer completed cohort."""
    spec = MetricSpec(
        name="ratio",
        type="ratio",
        numerator="num",
        denominator="den",
        window_days=3,
        missing="zero",
    )

    def build(denominator_end: int, observation_end: str | None = None) -> FramePanelSource:
        rows = []
        for unit, group, exposure in (
            ("c0", "control", 0),
            ("t0", "treatment", 0),
            ("c1", "control", 2),
            ("t1", "treatment", 2),
        ):
            for day in range(exposure, 6):
                rows.append(
                    (
                        unit,
                        group,
                        f"d{day}",
                        f"d{exposure}",
                        1.0,
                        1.0 if day <= denominator_end else None,
                    )
                )
        return from_unit_panel(
            pa.table(
                dict(
                    zip(
                        ["user_id", "variant", "day", "exposed_on", "num", "den"],
                        list(zip(*rows, strict=True)),
                        strict=True,
                    )
                )
            ),
            unit="user_id",
            group="variant",
            date="day",
            exposure_date="exposed_on",
            control="control",
            metrics=[spec],
            observation_end=observation_end,
        )

    lagging = build(2)
    lagging_rows = lagging.moments(
        _metric(lagging, "ratio"), grain="asof", completed_windows_only=True
    )
    assert {row["ds"] for row in lagging_rows} == {"d3", "d4", "d5"}
    assert {row["n"] for row in lagging_rows} == {1}
    assert {_sum_y(row) for row in lagging_rows} == {3.0}
    assert {_sum_den(row) for row in lagging_rows} == {3.0}
    partial = lagging.moments(_metric(lagging, "ratio"), grain="asof")
    latest_partial = [row for row in partial if row["ds"] == "d5"]
    assert {row["n"] for row in latest_partial} == {2}
    assert {_sum_y(row) for row in latest_partial} == {6.0}
    assert {_sum_den(row) for row in latest_partial} == {4.0}

    caught_up = build(4)
    caught_up_rows = caught_up.moments(
        _metric(caught_up, "ratio"), grain="asof", completed_windows_only=True
    )
    latest = [row for row in caught_up_rows if row["ds"] == "d5"]
    assert {row["n"] for row in latest} == {2}
    assert {_sum_y(row) for row in latest} == {6.0}
    assert {_sum_den(row) for row in latest} == {6.0}
    declared = build(2, observation_end="d5")
    declared_rows = declared.moments(
        _metric(declared, "ratio"), grain="asof", completed_windows_only=True
    )
    latest_declared = [row for row in declared_rows if row["ds"] == "d5"]
    assert {row["n"] for row in latest_declared} == {2}
    assert {_sum_y(row) for row in latest_declared} == {6.0}
    assert {_sum_den(row) for row in latest_declared} == {4.0}


@pytest.mark.parametrize("absent", ["num", "den"])
def test_ratio_all_null_component_keeps_zero_filled_units(absent: str) -> None:
    rows = [
        (unit, group, day, 0, None if absent == "num" else 1.0, None if absent == "den" else 1.0)
        for unit, group in (("c", "control"), ("t", "treatment"))
        for day in range(4)
    ]
    table = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "exposed_on", "num", "den"],
                zip(*rows, strict=True),
                strict=True,
            )
        )
    )
    source = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        exposure_date="exposed_on",
        control="control",
        metrics=[
            MetricSpec(
                name="ratio",
                type="ratio",
                numerator="num",
                denominator="den",
                window_days=3,
                missing="zero",
            )
        ],
    )
    metric = _metric(source, "ratio")
    for moments in (
        source.moments(metric),
        source.moments(metric, grain="asof", completed_windows_only=True),
    ):
        assert {row["group_id"] for row in moments} == {"control", "treatment"}
        assert {row["n"] for row in moments} == {1}
        assert {_sum_y(row) for row in moments} == {0.0 if absent == "num" else 3.0}
        assert {_sum_den(row) for row in moments} == {0.0 if absent == "den" else 3.0}
