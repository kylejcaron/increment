"""Tests for the dataframe entry point (``increment/frame.py``).

The load-bearing tests are ``test_moments_match_group_summary`` and
``test_daily_moments_match_daily_group_summary``: they assert the narwhals
moments equal the real ibis builders column for column. Both import ibis;
``increment/frame.py`` must not.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, cast

import pyarrow as pa
import pytest

from increment import Analysis
from increment.errors import CapabilityError, InvalidRequestError
from increment.frame import (
    FramePanelSource,
    FrameTotalsSource,
    MetricSpec,
    from_unit_panel,
    from_unit_summary,
)
from increment.semantics.models import Metric
from tests.analysis_factory import lift_rows


def _sum_y(row: Any) -> float:
    """Recover sum(y) from a moments row: n*ref_y + cy1, exact and
    cancellation-free."""
    return row["n"] * row["ref_y"] + row["cy1"]


def _sum_yd(row: Any) -> float:
    """Recover sum(d*y) from a moments row: cyd + sum_d*ref_y."""
    return row["cyd"] + row["sum_d"] * row["ref_y"]


def _sum_y2d(row: Any) -> float:
    """Recover sum(d*y**2): cy2d + 2*ref_y*cyd + sum_d*ref_y**2."""
    return row["cy2d"] + 2.0 * row["ref_y"] * row["cyd"] + row["sum_d"] * row["ref_y"] ** 2


_ROWS = [
    # unit,  variant,     revenue, converted, pre_revenue, orders, pre_balanced
    ("u01", "control", 22.40, 1.0, 9.10, 3.0, 16.3333),
    ("u02", "control", 10.00, 0.0, 3.75, 1.0, 3.9333),
    ("u03", "control", 41.05, 1.0, 28.60, 5.0, 34.9833),
    ("u04", "control", 14.20, 1.0, 5.05, 2.0, 8.1333),
    ("u05", "control", 10.00, 0.0, 0.00, 1.0, 3.9333),
    ("u06", "control", 28.75, 1.0, 15.20, 4.0, 22.6833),
    ("u07", "treatment", 32.10, 1.0, 8.90, 4.0, 19.3000),
    ("u08", "treatment", 15.60, 1.0, 4.10, 2.0, 2.8000),
    ("u09", "treatment", 51.30, 1.0, 30.15, 7.0, 38.5000),
    ("u10", "treatment", 10.00, 0.0, 2.20, 1.0, -2.8000),
    ("u11", "treatment", 37.85, 1.0, 16.40, 5.0, 25.0500),
    ("u12", "treatment", 19.95, 1.0, 6.75, 3.0, 7.1500),
]


_COLUMNS = [
    "user_id",
    "variant",
    "revenue",
    "converted",
    "pre_revenue",
    "orders",
    "pre_balanced",
]


def _arrow_table() -> pa.Table:
    cols = list(zip(*_ROWS, strict=True))
    return pa.table(dict(zip(_COLUMNS, cols, strict=True)))


@pytest.fixture
def arrow_frame() -> pa.Table:
    return _arrow_table()


def _metric(src: FrameTotalsSource | FramePanelSource, name: str) -> Metric:
    """Typed lookup into ``src.metrics`` - the protocol types it
    ``Sequence[object]`` (zero-dependency seam), so tests that need
    ``.name`` narrow it back explicitly rather than fighting ty at every
    call site."""
    return next(m for m in src.context.metrics if m.name == name)


def test_from_unit_summary_uptake_moments_match_hand_computation() -> None:
    """4 units per arm, hand-computable sums - mirrors the ArmStats fixture
    in ``tests/estimation/test_armstats.py`` (``TestUptakeReductions``).

    The uptake family is centered on the arm's OVERALL reference ref_y, not
    on the takers' own mean:
      treatment y = [1, 2, 3, 4], d = [0, 1, 1, 0], ref_y = 2.5
        -> sum_d=2, cyd=(2-2.5)+(3-2.5)=0, cy2d=0.25+0.25=0.5
      control (mirror image) y = [5, 6, 7, 8], d = [1, 0, 0, 1], ref_y = 6.5
        -> sum_d=2, cyd=(5-6.5)+(8-6.5)=0, cy2d=2.25+2.25=4.5

    Both arms' takers straddle their reference exactly, so cyd cancels to a
    true zero here; the discriminating numbers are cy2d and the classic sums
    the analytic expansions recover from them (sum_yd = 13 / 5,
    sum_y2d = 89 / 13), which is what this fixture always pinned.
    """
    rows = [
        ("u01", "control", 5.0, 1),
        ("u02", "control", 6.0, 0),
        ("u03", "control", 7.0, 0),
        ("u04", "control", 8.0, 1),
        ("u05", "treatment", 1.0, 0),
        ("u06", "treatment", 2.0, 1),
        ("u07", "treatment", 3.0, 1),
        ("u08", "treatment", 4.0, 0),
    ]
    cols = list(zip(*rows, strict=True))
    table = pa.table(dict(zip(["user_id", "variant", "revenue", "clicked"], cols, strict=True)))

    src = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        uptake="clicked",
    )
    by_group = {r["group_id"]: r for r in src.raw_moments}
    control = by_group["control"]
    treatment = by_group["treatment"]

    assert control["sum_d"] == pytest.approx(2.0)
    assert control["cyd"] == pytest.approx(0.0, abs=1e-12)
    assert control["cy2d"] == pytest.approx(4.5)
    assert _sum_yd(control) == pytest.approx(13.0)
    assert _sum_y2d(control) == pytest.approx(89.0)
    assert treatment["sum_d"] == pytest.approx(2.0)
    assert treatment["cyd"] == pytest.approx(0.0, abs=1e-12)
    assert treatment["cy2d"] == pytest.approx(0.5)
    assert _sum_yd(treatment) == pytest.approx(5.0)
    assert _sum_y2d(treatment) == pytest.approx(13.0)


def test_from_unit_summary_encouragement_end_to_end() -> None:
    """from_unit_summary(uptake=...) feeding estimate_encouragement returns
    itt, compliance, and late rows - the full path from a raw dataframe to
    an Encouragement readout, entirely through frame.py's own public surface
    (from_unit_summary + the already-wired estimate_encouragement/ArmStats
    seam), independent of whether the higher-level Analysis/readouts facade
    has been wired to the encouragement mechanism yet.
    """
    from increment.estimation.encouragement import estimate_encouragement
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    rows = [
        ("u01", "control", 10.0, 0),
        ("u02", "control", 12.0, 0),
        ("u03", "control", 9.0, 0),
        ("u04", "control", 20.0, 1),
        ("u05", "treatment", 25.0, 1),
        ("u06", "treatment", 30.0, 1),
        ("u07", "treatment", 28.0, 1),
        ("u08", "treatment", 15.0, 0),
    ]
    cols = list(zip(*rows, strict=True))
    table = pa.table(dict(zip(["user_id", "variant", "revenue", "clicked"], cols, strict=True)))

    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
        # A low z-floor keeps LATE emitted on this small, imperfect-compliance
        # fixture (4 units/arm) instead of exercising the default weak-instrument suppression path.
        min_first_stage_z=0.5,
    )
    src = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        uptake="clicked",
        design=design,
    )
    metrics = cast("list[Metric]", src.context.metrics)
    results = estimate_encouragement(metrics, src.raw_moments, design).results
    assert {r.estimand for r in results} == {"itt", "compliance", "late"}


def test_uptake_column_must_be_binary() -> None:
    """d = 0.5 is refused by code: an encouragement design's uptake column
    must be exactly 0/1."""
    rows = [
        ("u01", "control", 5.0, 0.5),
        ("u02", "control", 6.0, 0.0),
        ("u03", "treatment", 1.0, 1.0),
        ("u04", "treatment", 2.0, 0.0),
    ]
    cols = list(zip(*rows, strict=True))
    table = pa.table(dict(zip(["user_id", "variant", "revenue", "clicked"], cols, strict=True)))

    with pytest.raises(InvalidRequestError) as exc:
        from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
            uptake="clicked",
        )
    assert exc.value.code == "source.frame.uptake_not_binary"


def test_encouragement_uptake_column_derives_from_the_design() -> None:
    """The design already names its uptake fact; on a dataframe the fact IS
    the column, so uptake= is redundant. Deriving it must produce exactly
    the moments an explicit uptake= produces."""
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    rows = [
        ("u01", "control", 5.0, 1.0),
        ("u02", "control", 6.0, 0.0),
        ("u03", "treatment", 1.0, 0.0),
        ("u04", "treatment", 2.0, 1.0),
    ]
    cols = list(zip(*rows, strict=True))
    table = pa.table(dict(zip(["user_id", "variant", "revenue", "clicked"], cols, strict=True)))
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
    )
    common: dict[str, Any] = {
        "unit": "user_id",
        "group": "variant",
        "control": "control",
        "metrics": {"revenue": "mean"},
        "design": design,
    }

    derived = from_unit_summary(table, **common)
    explicit = from_unit_summary(table, **common, uptake="clicked")

    assert derived.raw_moments == explicit.raw_moments


def test_explicit_uptake_column_overrides_the_declared_fact() -> None:
    """uptake= is the override, mirroring MetricSpec.value_column over name:
    when the two disagree, the kwarg names the column that is analyzed."""
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    # 'clicked' (what the design declares) is all zeros; 'took' (the
    # override) is all ones, so sum_d alone proves which column was read.
    rows = [
        ("u01", "control", 5.0, 0.0, 1.0),
        ("u02", "control", 6.0, 0.0, 1.0),
        ("u03", "treatment", 1.0, 0.0, 1.0),
        ("u04", "treatment", 2.0, 0.0, 1.0),
    ]
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(zip(["user_id", "variant", "revenue", "clicked", "took"], cols, strict=True))
    )
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
    )

    src = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        design=design,
        uptake="took",
    )

    for row in src.raw_moments:
        assert row["sum_d"] == pytest.approx(row["n"])


def test_derived_uptake_column_missing_from_frame_names_its_source(
    arrow_frame: pa.Table,
) -> None:
    """A design whose fact does not name a column must say so, and say the
    design is where the name came from - the old error told callers to pass
    uptake=, which is no longer the requirement."""
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
    )
    with pytest.raises(InvalidRequestError) as exc:
        from_unit_summary(
            arrow_frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
            design=design,
        )
    assert exc.value.code == "source.frame.metric_missing"
    # The missing-column report names the column and the declaration the name
    # came from, so a caller learns the design owns it rather than being told
    # to pass uptake=.
    missing = cast("list[str]", exc.value.context["missing"])
    assert any("'clicked'" in entry and "design.uptake.fact" in entry for entry in missing)


def test_from_unit_panel_collapses_uptake_ever() -> None:
    """A unit with a day-5 click gets d=1 at 'total' grain (no window set)."""
    days = [1, 2, 3, 4, 5]
    click_day = {"u1": 5, "u2": None, "u3": None, "u4": None}
    group = {"u1": "treatment", "u2": "treatment", "u3": "control", "u4": "control"}
    rows = []
    for day in days:
        for unit in ("u1", "u2", "u3", "u4"):
            clicked = 1.0 if day == click_day[unit] else 0.0
            rows.append((unit, group[unit], day, 1.0, clicked))
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(zip(["user_id", "variant", "day", "revenue", "clicked"], cols, strict=True))
    )

    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        uptake="clicked",
    )
    metric = _metric(src, "revenue")
    by_group = {r["group_id"]: r for r in src.moments(metric, grain="total")}
    assert by_group["treatment"]["sum_d"] == pytest.approx(1.0)
    assert by_group["control"]["sum_d"] == pytest.approx(0.0)


def test_from_unit_panel_derives_uptake_column_alongside_the_window() -> None:
    """The panel path reads design.uptake for BOTH the derived column and
    window_days; deriving the column must not disturb the window scoping.

    Unlike test_from_unit_panel_uptake_window's fixture, this one has no
    day-0 row: each unit's first observed day is day 1, so elapsed =
    day - 1. The window is still load-bearing (u1's day-5 click is
    elapsed 4, outside [0, 3); u2's day-2 click is elapsed 1, inside).
    """
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    click_day = {"u1": 5, "u2": 2, "u3": None, "u4": None}
    group = {"u1": "treatment", "u2": "treatment", "u3": "control", "u4": "control"}
    rows = []
    for day in (1, 2, 3, 4, 5):
        for unit in ("u1", "u2", "u3", "u4"):
            clicked = 1.0 if day == click_day[unit] else 0.0
            rows.append((unit, group[unit], day, 1.0, clicked))
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(zip(["user_id", "variant", "day", "revenue", "clicked"], cols, strict=True))
    )
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked", window_days=3),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
    )
    common: dict[str, Any] = {
        "unit": "user_id",
        "group": "variant",
        "date": "day",
        "control": "control",
        "metrics": {"revenue": "mean"},
        "design": design,
    }

    derived = from_unit_panel(table, **common)
    explicit = from_unit_panel(table, **common, uptake="clicked")

    derived_rows = derived.moments(_metric(derived, "revenue"), grain="total")
    explicit_rows = explicit.moments(_metric(explicit, "revenue"), grain="total")
    assert derived_rows == explicit_rows
    # The window still bites: only u2's in-window click counts for treatment.
    by_group = {r["group_id"]: r for r in derived_rows}
    assert by_group["treatment"]["sum_d"] == pytest.approx(1.0)


def test_from_unit_panel_uptake_window() -> None:
    """window_days=3, click on day 5 -> d=0; click on day 2 -> d=1.

    Both units share first exposure at day 0 (their earliest panel row), so
    day values double as elapsed-since-exposure offsets: day 5 falls outside
    the half-open [0, 3) window, day 2 falls inside it. Without the window
    (design=None) both clicks count, so the aggregate contrast (sum_d=1 vs
    sum_d=2) pins the windowed behaviour precisely.
    """
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    days = [0, 1, 2, 3, 4, 5]
    click_day = {"u1": 5, "u2": 2, "u3": None, "u4": None}
    group = {"u1": "treatment", "u2": "treatment", "u3": "control", "u4": "control"}
    rows = []
    for day in days:
        for unit in ("u1", "u2", "u3", "u4"):
            clicked = 1.0 if day == click_day[unit] else 0.0
            rows.append((unit, group[unit], day, 1.0, clicked))
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(zip(["user_id", "variant", "day", "revenue", "clicked"], cols, strict=True))
    )

    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked", window_days=3),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
    )
    windowed = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        uptake="clicked",
        design=design,
    )
    metric = _metric(windowed, "revenue")
    windowed_treatment = {r["group_id"]: r for r in windowed.moments(metric, grain="total")}[
        "treatment"
    ]
    assert windowed_treatment["sum_d"] == pytest.approx(1.0)  # only u2's day-2 click counts

    unwindowed = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        uptake="clicked",
    )
    metric_uw = _metric(unwindowed, "revenue")
    unwindowed_treatment = {r["group_id"]: r for r in unwindowed.moments(metric_uw, grain="total")}[
        "treatment"
    ]
    assert unwindowed_treatment["sum_d"] == pytest.approx(2.0)  # both u1 and u2 count


def test_from_unit_panel_uptake_window_anchors_to_each_units_own_entry() -> None:
    """Staggered enrollment: window anchors to each unit's OWN first
    observed day, never the panel's global earliest date.

    u1 enters at day 0 (rows for days 0-5); u2 enters at day 3 (rows only
    from day 3 - days 0-2 are absent from the INPUT and only appear after
    densification, zero-filled). window_days=3.

    u1 clicks day 1 -> elapsed since u1's own entry (day 0) is 1, inside
    [0, 3) -> counts. u2 clicks day 4 -> elapsed since u2's OWN entry
    (day 3) is 1, inside [3, 6) -> counts under correct per-unit
    anchoring. Anchoring to the panel's GLOBAL earliest date (day 0, from
    the densified spine) instead would evaluate u2's day-4 click as
    elapsed=4, outside [0, 3) - wrongly dropped. sum_d=2 (both count)
    only holds under correct per-unit anchoring; the bug would yield 1.
    """
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    rows = []
    for day in (0, 1, 2, 3, 4, 5):
        clicked = 1.0 if day == 1 else 0.0
        rows.append(("u1", "treatment", day, 1.0, clicked))
    for day in (3, 4, 5):  # u1 enters at 0, u2 enters at 3 - staggered entry
        clicked = 1.0 if day == 4 else 0.0
        rows.append(("u2", "treatment", day, 1.0, clicked))
    for day in (0, 1, 2, 3, 4, 5):
        rows.append(("u3", "control", day, 1.0, 0.0))
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(zip(["user_id", "variant", "day", "revenue", "clicked"], cols, strict=True))
    )

    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked", window_days=3),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        uptake="clicked",
        design=design,
    )
    metric = _metric(src, "revenue")
    treatment = {r["group_id"]: r for r in src.moments(metric, grain="total")}["treatment"]
    assert treatment["sum_d"] == pytest.approx(2.0), (
        "both u1's day-1 click and u2's day-4 click must count under "
        "per-unit anchoring; a global-anchor bug would drop u2's"
    )


def test_from_unit_panel_uptake_window_with_real_dates() -> None:
    """Same windowed-collapse logic on an actual date column, not bare ints.

    Exercises the Date/Datetime elapsed-time branch (duration subtraction
    + total_seconds()/86400), which an all-integer-day fixture never
    reaches.
    """
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    base = date(2025, 1, 1)
    offsets = {"u1": 5, "u2": 2}
    group = {"u1": "treatment", "u2": "treatment"}
    rows = []
    for offset in range(6):
        day = base + timedelta(days=offset)
        for unit in ("u1", "u2"):
            clicked = 1.0 if offset == offsets[unit] else 0.0
            rows.append((unit, group[unit], day, 1.0, clicked))
    rows.append(("u3", "control", base, 1.0, 0.0))
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(zip(["user_id", "variant", "day", "revenue", "clicked"], cols, strict=True))
    )

    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked", window_days=3),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        uptake="clicked",
        design=design,
    )
    metric = _metric(src, "revenue")
    treatment = {r["group_id"]: r for r in src.moments(metric, grain="total")}["treatment"]
    assert treatment["sum_d"] == pytest.approx(1.0)  # only u2's day-2 (offset) click counts


def _encouragement_design(window_days: int | None = None):
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    return Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked", window_days=window_days),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
    )


def test_metric_free_fixed_horizon_compliance_matches_populated_oracle() -> None:
    """Summary and panel frames keep the design-only first stage metric-free."""
    rows = [
        ("c1", "control", 0.0, 10.0),
        ("c2", "control", 0.0, 12.0),
        ("t1", "treatment", 1.0, 20.0),
        ("t2", "treatment", 0.0, 22.0),
    ]
    table = pa.table(
        dict(
            zip(
                ["user_id", "variant", "clicked", "revenue"],
                list(zip(*rows, strict=True)),
                strict=True,
            )
        )
    )
    design = _encouragement_design()

    populated = lift_rows(
        Analysis.from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            metrics={"revenue": "mean"},
            design=design,
        ).run(estimands=("compliance",))
    )
    empty = lift_rows(
        Analysis.from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            metrics=[],
            design=design,
        ).run(estimands=("compliance",))
    )
    populated = [result for result in populated if result.value_scale == "absolute"]
    empty = [result for result in empty if result.value_scale == "absolute"]
    assert len(empty) == len(populated) == 1
    actual, expected = empty[0].lift, populated[0].lift
    assert actual is not None and expected is not None
    assert actual.value == pytest.approx(0.5)
    assert (actual.value, actual.lb, actual.ub) == pytest.approx(
        (expected.value, expected.lb, expected.ub)
    )
    summary_source = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[],
        design=design,
    )
    for group in ("control", "treatment"):
        arm = summary_source.compliance_summary(design).arm(group)
        assert arm is not None
        assert arm.n_units == 2

    panel = pa.table(
        {
            "user_id": [row[0] for row in rows],
            "variant": [row[1] for row in rows],
            "day": [0] * len(rows),
            "clicked": [row[2] for row in rows],
            "revenue": [row[3] for row in rows],
        }
    )
    panel_populated = lift_rows(
        Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="day",
            metrics={"revenue": "mean"},
            design=design,
        ).run(estimands=("compliance",))
    )
    panel_empty = lift_rows(
        Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="day",
            metrics=[],
            design=design,
        ).run(estimands=("compliance",))
    )
    panel_populated = [result for result in panel_populated if result.value_scale == "absolute"]
    panel_empty = [result for result in panel_empty if result.value_scale == "absolute"]
    assert len(panel_empty) == len(panel_populated) == 1
    actual, expected = panel_empty[0].lift, panel_populated[0].lift
    assert actual is not None and expected is not None
    assert actual.value == pytest.approx(0.5)
    assert (actual.value, actual.lb, actual.ub) == pytest.approx(
        (expected.value, expected.lb, expected.ub)
    )
    panel_source = from_unit_panel(
        panel,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[],
        design=design,
    )
    panel_summary = panel_source.compliance_summary(design)
    for group in ("control", "treatment"):
        arm = panel_summary.arm(group)
        assert arm is not None
        assert arm.n_units == 2


def _pre_exposure_click_rows() -> list[tuple[str, str, date, date, float, float]]:
    """u1 (treatment): panel days 0..5, explicit ``exposed_on`` = day 3, its
    only click on day 0 - three days BEFORE its own exposure. u2
    (control): single day-0 row, no clicks."""
    base = date(2025, 1, 1)
    rows = []
    for off in range(6):
        clicked = 1.0 if off == 0 else 0.0
        rows.append(
            ("u1", "treatment", base + timedelta(days=off), base + timedelta(days=3), 1.0, clicked)
        )
    rows.append(("u2", "control", base, base, 1.0, 0.0))
    return rows


@pytest.mark.parametrize("window_days", [2, None], ids=["windowed", "ever"])
def test_uptake_band_excludes_pre_exposure_clicks(window_days: int | None) -> None:
    """A click BEFORE the unit's own exposure never counts as compliance:
    the uptake band is ``[first_exposure, first_exposure + window_days)``
    (or ``[first_exposure, ∞)`` unwindowed) - the same unconditional
    ``ts >= first_exposure_ts`` lower bound builders.unit_totals applies.
    Counting it inflates the first stage and biases LATE toward zero.
    """
    rows = _pre_exposure_click_rows()
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "exposed_on", "revenue", "clicked"],
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
        metrics={"revenue": "mean"},
        uptake="clicked",
        design=_encouragement_design(window_days),
        exposure_date="exposed_on",
    )
    metric = _metric(src, "revenue")
    treatment = {r["group_id"]: r for r in src.moments(metric, grain="total")}["treatment"]
    assert treatment["sum_d"] == pytest.approx(0.0)


def test_uptake_window_pandas_object_dates_does_not_crash() -> None:
    """pandas stores ``datetime.date`` columns as object dtype, which
    narwhals does not recognise as Date/Datetime - the uptake-window
    elapsed arithmetic must cast both sides (as ``_with_day_index`` does)
    instead of comparing a raw timedelta against an int (``TypeError``).
    The band itself must also hold: u1's pre-exposure click reads 0, u3's
    day-after-exposure click reads 1.
    """
    pd = pytest.importorskip("pandas")

    base = date(2025, 1, 1)
    rows = _pre_exposure_click_rows()
    # u3: exposed day 0, clicks day 1 - inside the 2-day band.
    rows.append(("u3", "treatment", base, base, 1.0, 0.0))
    rows.append(("u3", "treatment", base + timedelta(days=1), base, 1.0, 1.0))
    frame = pd.DataFrame(
        rows, columns=["user_id", "variant", "day", "exposed_on", "revenue", "clicked"]
    )
    assert frame["day"].dtype == object  # genuine datetime.date objects

    src = from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        uptake="clicked",
        design=_encouragement_design(window_days=2),
        exposure_date="exposed_on",
    )
    metric = _metric(src, "revenue")
    treatment = {r["group_id"]: r for r in src.moments(metric, grain="total")}["treatment"]
    assert treatment["sum_d"] == pytest.approx(1.0)  # u3 only; u1's click is pre-exposure


def test_plain_conversion_refuses_day_count_collapse() -> None:
    """A conversion is ANY-OCCURRENCE (0/1), always. The unwindowed panel
    collapse sums per-day rows, so a unit converting on 3 days would
    silently read ``y=3`` - a day count wearing a conversion metric's
    name. Refuse, pointing at a count aggregation for the day-count
    question.
    """
    base = date(2025, 1, 1)
    rows = [
        ("u1", "treatment", base + timedelta(days=off), 1.0 if off < 3 else 0.0) for off in range(5)
    ]
    rows.append(("u2", "control", base, 0.0))
    table = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "converted"],
                list(zip(*rows, strict=True)),
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
        metrics=[MetricSpec(name="converted", type="conversion")],
    )
    metric = _metric(src, "converted")
    with pytest.raises(InvalidRequestError) as exc:
        src.moments(metric, grain="total")
    assert exc.value.code == "frame.moments.metric_type_conversion"


def test_plain_conversion_single_converting_day_is_served() -> None:
    """At most one converting day per unit keeps the collapse inside {0,1}
    - genuinely Bernoulli, so it is served, not refused."""
    base = date(2025, 1, 1)
    rows = [
        ("u1", "treatment", base + timedelta(days=off), 1.0 if off == 2 else 0.0)
        for off in range(5)
    ]
    rows.append(("u2", "control", base, 0.0))
    table = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "converted"],
                list(zip(*rows, strict=True)),
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
        metrics=[MetricSpec(name="converted", type="conversion")],
    )
    metric = _metric(src, "converted")
    out = {r["group_id"]: r for r in src.moments(metric, grain="total")}
    assert _sum_y(out["treatment"]) == pytest.approx(1.0)
    assert _sum_y(out["control"]) == pytest.approx(0.0)


def test_from_unit_panel_refuses_declared_retention_metric_under_encouragement() -> None:
    """A retention metric's maturity gate changes the first-stage population
    by outcome metric -- refused with the same code the readout layer uses
    for every other constructor, caught while scanning the raw declared
    metrics, before ``coerce_metrics`` runs."""
    table = pa.table({"user_id": ["u1", "u2"], "variant": ["treatment", "control"], "day": [0, 0]})
    design = _encouragement_design()
    with pytest.raises(CapabilityError) as exc_info:
        from_unit_panel(
            table,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[MetricSpec(name="d7", type="retention", threshold_days=7)],
            design=design,
        )
    assert exc_info.value.code == "readout.encouragement.retention"
    assert exc_info.value.context == {"names": ("d7",)}
