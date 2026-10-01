"""Tests for the dataframe entry point (``increment/frame.py``).

The load-bearing tests are ``test_moments_match_group_summary`` and
``test_daily_moments_match_daily_group_summary``: they assert the narwhals
moments equal the real ibis builders column for column. Both import ibis;
``increment/frame.py`` must not.
"""

from __future__ import annotations

import importlib.util
import math
from datetime import date, timedelta
from typing import Any

import narwhals as nw
import numpy as np
import pyarrow as pa
import pytest

from increment import readouts
from increment.errors import (
    CapabilityError,
    IncrementWarning,
    InvalidRequestError,
    UnsupportedRequestError,
)
from increment.estimation.diagnostics import SRMResult
from increment.estimation.engine import Method
from increment.frame import (
    FramePanelSource,
    FrameTotalsSource,
    MetricSpec,
    from_unit_panel,
    from_unit_summary,
)
from increment.semantics.design import Randomized
from increment.semantics.models import Metric
from tests.warning_codes import warning_codes

_PLUGIN_TOL = 1e-6


_DESIGN = Randomized(
    control_group="control",
    allocation={"control": 0.5, "treatment": 0.5},
)


def _sum_y(row: Any) -> float:
    """Recover sum(y) from a moments row: n*ref_y + cy1, exact and
    cancellation-free."""
    return row["n"] * row["ref_y"] + row["cy1"]


def _sum_x(row: Any) -> float:
    """Recover sum(x) from a moments row: n*ref_x + cx1."""
    return row["n"] * row["ref_x"] + row["cx1"]


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


@pytest.fixture
def panel_frame() -> pa.Table:
    return _panel_table()


def _encouragement_design(window_days: int | None = None):
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    return Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked", window_days=window_days),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
    )


def _fence_panel(observation_end: date) -> FramePanelSource:
    """u_late (treatment) exposed 2025-01-08 with window ``[0, 7)`` = dates
    01-08..01-14, revenue 2.0/day; u_early (control) exposed 01-01,
    revenue 1.0/day. Panel observed 01-01..01-14."""
    base = date(2025, 1, 1)
    rows = []
    for off in range(14):
        d0 = base + timedelta(days=off)
        rows.append(("u_late", "treatment", d0, date(2025, 1, 8), 2.0))
        rows.append(("u_early", "control", d0, base, 1.0))
    table = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "exposed_on", "revenue"],
                list(zip(*rows, strict=True)),
                strict=True,
            )
        )
    )
    return from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", type="mean", window_days=7)],
        exposure_date="exposed_on",
        observation_end=observation_end,
    )


def test_censoring_fence_keeps_fully_observed_unit() -> None:
    """A unit whose ENTIRE window ``[0, 7)`` (dates 01-08..01-14) is
    observed through ``observation_end=01-14`` is kept: maturity compares
    the LAST window day against the observable bound, never the exclusive
    right edge - which dropped exactly the newest fully-observed cohort
    every day, forever."""
    src = _fence_panel(date(2025, 1, 14))
    metric = _metric(src, "revenue")
    out = {r["group_id"]: r for r in src.moments(metric, grain="total")}
    assert out["treatment"]["n"] == 1
    assert _sum_y(out["treatment"]) == pytest.approx(14.0)  # 7 days x 2.0
    assert out["control"]["n"] == 1
    assert _sum_y(out["control"]) == pytest.approx(7.0)


def test_censoring_fence_still_drops_genuinely_immature_unit() -> None:
    """One day short (observation_end=01-13, last window day 01-14): the
    same unit's outcome is NOT final and it is dropped whole."""
    src = _fence_panel(date(2025, 1, 13))
    metric = _metric(src, "revenue")
    with pytest.warns(IncrementWarning) as rec:
        out = {r["group_id"]: r for r in src.moments(metric, grain="total")}
    assert "frame.censoring.dropped_units" in warning_codes(rec)
    assert "treatment" not in out or out["treatment"]["n"] == 0
    assert out["control"]["n"] == 1


def _null_group_table() -> pa.Table:
    """The phantom-arm shape: 198 control, 182 treatment, 20 null labels."""
    labels = ["control"] * 198 + ["treatment"] * 182 + [None] * 20
    return pa.table(
        {
            "user_id": [f"u{i:03d}" for i in range(400)],
            "variant": labels,
            "revenue": [float((i % 7) + 1) for i in range(400)],
        }
    )


def test_unit_counts_on_a_plain_summary_source() -> None:
    """The unclustered baseline: one key per arm, no accounting keys, and
    cluster_counts() refuses because no randomization grain was declared."""
    labels = ["control"] * 198 + ["treatment"] * 182
    src = from_unit_summary(
        pa.table(
            {
                "user_id": [f"u{i:03d}" for i in range(380)],
                "variant": labels,
                "revenue": [float((i % 7) + 1) for i in range(380)],
            }
        ),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    )
    assert src.unit_counts() == {"control": 198, "treatment": 182}
    with pytest.raises(CapabilityError) as raised:
        src.cluster_counts()
    assert raised.value.code == "source.frame.cluster_grain"


def test_unit_counts_ignore_first_metric_drop_and_declaration_order() -> None:
    """unit_counts() is the enrolled population, not the first metric's
    complete-case count: a metric with missing='drop' rows the others lack
    must not shrink the sample-ratio input, and reversing the declared
    metric order must not move it."""
    orders = [float((i % 5) + 1) for i in range(200)]  # fully observed
    table = _half_null_table().append_column("orders", pa.array(orders, type=pa.float64()))
    kwargs: dict[str, Any] = {
        "unit": "user_id",
        "group": "variant",
        "control": "control",
    }
    drop_first = [
        MetricSpec(name="revenue", missing="drop"),
        MetricSpec(name="orders", value_column="orders"),
    ]
    with pytest.warns(IncrementWarning) as rec_first:
        src_drop_first = from_unit_summary(table, metrics=drop_first, **kwargs)
    assert "frame.validation.metric_missing_drop" in warning_codes(rec_first)
    with pytest.warns(IncrementWarning) as rec_last:
        src_drop_last = from_unit_summary(table, metrics=list(reversed(drop_first)), **kwargs)
    assert "frame.validation.metric_missing_drop" in warning_codes(rec_last)

    enrolled = {"control": 100, "treatment": 100}
    assert src_drop_first.unit_counts() == enrolled
    assert src_drop_last.unit_counts() == enrolled
    # The dropped metric's own moments still shrink; only unit_counts is enrolled.
    revenue = _metric(src_drop_first, "revenue")
    moments = {r["group_id"]: r["n"] for r in src_drop_first.moments(revenue)}
    assert moments == {"control": 100, "treatment": 50}


def test_null_group_label_refuses_naming_the_exclude_knob() -> None:
    with pytest.raises(InvalidRequestError) as error:
        from_unit_summary(
            _null_group_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
        )
    assert error.value.code == "source.frame.unassigned"
    assert error.value.context["constructor"] == "from_unit_summary"
    assert error.value.context["group"] == "variant"


def test_null_group_label_excluded_with_accounting() -> None:
    """on_unassigned='exclude': no phantom arm, no estimate for it, no SRM
    degree of freedom - the 20 excluded units are surfaced in
    unit_counts() and on SRMResult.unassigned_units instead."""
    src = from_unit_summary(
        _null_group_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        on_unassigned="exclude",
        design=_DESIGN,
    )
    assert src.unit_counts() == {"control": 198, "treatment": 182, "(unassigned)": 20}

    estimates = readouts.run(src)
    assert [e.group_id for e in estimates] == ["treatment"]

    srm = readouts.srm(src)
    assert isinstance(srm, SRMResult)
    assert srm.df == 1  # two arms; the unassigned bucket earns no df
    assert srm.unassigned_units == 20
    assert set(srm.observed) == {"control", "treatment"}
    assert not srm.is_srm  # 198 vs 182 is unremarkable; 3-way was chi2=145.5


def test_null_group_label_panel_refuses_and_excludes() -> None:
    base = date(2025, 1, 1)
    units = [f"u{i}" for i in range(12)]
    labels: dict[str, str | None] = {
        u: ("treatment" if i % 2 else "control") for i, u in enumerate(units)
    }
    labels["u0"] = None  # every row of u0 lacks an assignment
    table = pa.table(
        {
            "user_id": [u for u in units for _ in range(2)],
            "variant": [labels[u] for u in units for _ in range(2)],
            "day": [base + timedelta(days=k) for _ in units for k in range(2)],
            "revenue": [float(k) for _ in units for k in range(2)],
        }
    )
    kwargs: dict[str, Any] = {
        "unit": "user_id",
        "group": "variant",
        "date": "day",
        "control": "control",
        "metrics": {"revenue": "mean"},
    }
    with pytest.raises(InvalidRequestError) as error:
        from_unit_panel(table, **kwargs)
    assert error.value.code == "source.frame.unassigned"
    src = from_unit_panel(table, **kwargs, on_unassigned="exclude")
    assert src.unit_counts() == {"control": 5, "treatment": 6, "(unassigned)": 1}


def test_null_group_label_partial_row_excludes_whole_unit_with_accounting() -> None:
    """A unit with a valid group label on one day and a null label on
    another must lose its arm and its outcome entirely - not keep the
    arm while the null-labelled row silently densifies to zero with no
    accounting. Regression: excluded_units was 0 for this shape because
    only the null-labelled ROW was dropped while the unit's other row
    kept it inside its arm."""
    base = date(2025, 1, 1)
    table = pa.table(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "variant": ["control", None, "treatment", "treatment", "control", "control"],
            "day": [base, base + timedelta(days=1)] * 3,
            "revenue": [5.0, 9.0, 1.0, 1.0, 2.0, 2.0],
        }
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        on_unassigned="exclude",
    )
    assert src.unit_counts() == {"control": 1, "treatment": 1, "(unassigned)": 1}
    moments = {r["group_id"]: r for r in src.moments(_metric(src, "revenue"))}
    assert moments["control"]["n"] == 1
    assert _sum_y(moments["control"]) == pytest.approx(4.0)  # u3 only; u1's 5.0 excluded whole


def _half_null_table() -> pa.Table:
    """The null-as-zero shape: half of treatment revenue missing."""
    revenue: list[float | None] = []
    for i in range(100):
        revenue.append(float((i % 9) + 1))  # control, fully observed
    for i in range(100):
        revenue.append(None if i < 50 else float((i % 9) + 3))
    return pa.table(
        {
            "user_id": [f"u{i:03d}" for i in range(200)],
            "variant": ["control"] * 100 + ["treatment"] * 100,
            "revenue": revenue,
        }
    )


def test_null_metric_value_refuses_naming_both_fixes() -> None:
    """The refusal names the declaration tier (missing='zero'/'drop') AND
    the helper tier (increment.impute) - every failure ships its fix."""
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            _half_null_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
        )
    assert exc_info.value.code == "frame.validation.metric_value_missing"
    assert exc_info.value.context["metric"] == "revenue"
    assert exc_info.value.context["column"] == "revenue"
    assert exc_info.value.context["n_missing"] == 50


def test_missing_zero_reproduces_null_as_zero_on_purpose() -> None:
    """missing='zero' is the old silent null-as-zero number as DECLARED
    semantics: lift equals the zero-fill prediction exactly, while
    missing='drop' gives the (different) complete-case answer."""
    table = _half_null_table()
    revenue = table["revenue"].to_pylist()
    t_vals = list(revenue[100:])
    c_vals = revenue[:100]
    zero_pred = (sum(v or 0.0 for v in t_vals) / 100) / (sum(c_vals) / 100) - 1
    observed_t = [v for v in t_vals if v is not None]
    cc_pred = (sum(observed_t) / len(observed_t)) / (sum(c_vals) / 100) - 1
    assert zero_pred != pytest.approx(cc_pred, rel=1e-3)  # the two semantics differ

    src_zero = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue", missing="zero")],
        design=_DESIGN,
    )
    (est_zero,) = readouts.run(src_zero)
    assert est_zero.require_lift().value == pytest.approx(zero_pred, rel=_PLUGIN_TOL)

    with pytest.warns(IncrementWarning) as rec:
        src_drop = from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="revenue", missing="drop")],
            design=_DESIGN,
        )
    assert "frame.validation.metric_missing_drop" in warning_codes(rec)
    (est_drop,) = readouts.run(src_drop)
    assert est_drop.require_lift().value == pytest.approx(cc_pred, rel=_PLUGIN_TOL)
    moments = {r["group_id"]: r for r in src_drop.moments(_metric(src_drop, "revenue"))}
    assert moments["treatment"]["n"] == 50  # complete-case n, visibly reduced


def test_missing_policy_is_per_spec_not_per_column() -> None:
    """Two metrics reading ONE column under different declarations get
    different moments - the policy is applied per spec at moment
    construction, never by mutating the shared frame."""
    table = _half_null_table()
    with pytest.warns(IncrementWarning) as rec:
        src = from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(name="rev_zero", value_column="revenue", missing="zero"),
                MetricSpec(name="rev_drop", value_column="revenue", missing="drop"),
            ],
        )
    assert "frame.validation.metric_missing_drop" in warning_codes(rec)
    zero = {r["group_id"]: r for r in src.moments(_metric(src, "rev_zero"))}
    drop = {r["group_id"]: r for r in src.moments(_metric(src, "rev_drop"))}
    assert zero["treatment"]["n"] == 100 and drop["treatment"]["n"] == 50
    assert _sum_y(zero["treatment"]) == pytest.approx(_sum_y(drop["treatment"]))
    zero_c = {k: v for k, v in zero["control"].items() if k != "metric"}
    drop_c = {k: v for k, v in drop["control"].items() if k != "metric"}
    assert zero_c == drop_c  # untouched arm identical, label aside


def test_inferred_all_null_metric_obeys_missing_policy() -> None:
    """An inferred all-null metric is nullable numeric data, not an
    unsupported dtype: zero keeps every unit as zero, while drop retains no
    units and error still refuses."""
    frames: list[Any] = [
        pa.table(
            {
                "unit": [1, 2, 3, 4],
                "variant": ["control", "control", "treatment", "treatment"],
                "y": [None, None, None, None],
            }
        ),
        pa.table(
            {
                "unit": [1, 2, 3, 4],
                "variant": ["control", "control", "treatment", "treatment"],
                "y": pa.array([None, None, None, None], type=pa.float64()),
            }
        ),
        pa.table(
            {
                "unit": [1, 2, 3, 4],
                "variant": ["control", "control", "treatment", "treatment"],
                "y": [0.0, 0.0, 0.0, 0.0],
            }
        ),
    ]
    if importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        frames.append(frames[0].to_pandas(types_mapper=pd.ArrowDtype))
    if importlib.util.find_spec("polars") is not None:
        import polars as pl

        frames.extend(
            [
                pl.DataFrame(
                    {
                        "unit": [1, 2, 3, 4],
                        "variant": ["control", "control", "treatment", "treatment"],
                        "y": [None, None, None, None],
                    }
                ),
                pl.DataFrame(
                    {
                        "unit": [1, 2, 3, 4],
                        "variant": ["control", "control", "treatment", "treatment"],
                        "y": pl.Series("y", [None, None, None, None], dtype=pl.Float64),
                    }
                ),
            ]
        )

    observed: list[list[tuple[str, int, float]]] = []
    for frame in frames:
        src = from_unit_summary(
            frame,
            unit="unit",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="y", missing="zero")],
        )
        observed.append(
            sorted(
                (str(row["group_id"]), row["n"], _sum_y(row))
                for row in src.moments(_metric(src, "y"))
            )
        )
    assert all(rows == observed[0] for rows in observed)
    assert observed[0] == [("control", 2, 0.0), ("treatment", 2, 0.0)]

    all_null = frames[0]
    with pytest.warns(IncrementWarning):
        src = from_unit_summary(
            all_null,
            unit="unit",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(name="y_zero", value_column="y", missing="zero"),
                MetricSpec(name="y_drop", value_column="y", missing="drop"),
            ],
        )
    zero = {row["group_id"]: row for row in src.moments(_metric(src, "y_zero"))}
    drop = list(src.moments(_metric(src, "y_drop")))
    assert zero["treatment"]["n"] == 2
    assert drop == []
    assert _sum_y(zero["treatment"]) == pytest.approx(0.0)

    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            all_null,
            unit="unit",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="y", missing="error")],
        )
    assert exc_info.value.code == "frame.validation.metric_value_missing"


def test_observed_binary_metric_remains_unsupported() -> None:
    frame = pa.table(
        {
            "unit": [1, 2, 3, 4],
            "variant": ["control", "control", "treatment", "treatment"],
            "y": [b"1", b"2", b"3", b"4"],
        }
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            frame,
            unit="unit",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="y", missing="zero")],
        )
    assert exc_info.value.code == "source.frame.column_dtype"


def test_all_null_fixed_size_binary_unknown_metric_refuses_stably() -> None:
    frame = pa.table(
        {
            "unit": [1, 2, 3, 4],
            "variant": ["control", "control", "treatment", "treatment"],
            "y": pa.array([None, None, None, None], type=pa.binary(4)),
        }
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            frame,
            unit="unit",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="y", missing="zero")],
        )
    assert exc_info.value.code == "source.frame.column_dtype"
    assert exc_info.value.context["column"] == "y"


def test_all_null_pandas_period_unknown_metric_refuses_stably() -> None:
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(
        {
            "unit": [1, 2, 3, 4],
            "variant": ["control", "control", "treatment", "treatment"],
            "y": pandas.Series([pandas.NaT] * 4, dtype="period[M]"),
        }
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            frame,
            unit="unit",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="y", missing="zero")],
        )
    assert exc_info.value.code == "source.frame.column_dtype"
    assert exc_info.value.context["column"] == "y"


@pytest.mark.parametrize("backend", ["arrow", "polars"])
def test_inferred_null_panel_preserves_fact_freshness(backend) -> None:
    start = date(2025, 1, 1)
    records = [
        {
            "unit": unit,
            "variant": arm,
            "day": start + timedelta(days=day),
            "exposure": start,
            "observed": 1.0 if day == 0 else None,
            "missing": None,
        }
        for unit, arm in enumerate(("control", "control", "treatment", "treatment"))
        for day in range(4)
    ]
    inferred = pa.Table.from_pylist(records)
    missing_column = inferred.schema.get_field_index("missing")
    typed = inferred.set_column(
        missing_column, "missing", pa.nulls(len(records), type=pa.float64())
    )
    zero = inferred.set_column(missing_column, "missing", pa.array([0.0] * len(records)))
    populations = []
    for frame in (inferred, typed, zero):
        native = pytest.importorskip("polars").from_arrow(frame) if backend == "polars" else frame
        source = from_unit_panel(
            native,
            unit="unit",
            group="variant",
            date="day",
            control="control",
            exposure_date="exposure",
            metrics=[
                MetricSpec(name="observed", window_days=2, missing="zero"),
                MetricSpec(name="missing", window_days=2, missing="zero"),
            ],
        )
        with pytest.warns(IncrementWarning) as warnings:
            missing_rows = source.moments(_metric(source, "missing"))
        populations.append(sum(row["n"] for row in missing_rows))
        assert "frame.censoring.dropped_units" in warning_codes(warnings)
        assert sum(row["n"] for row in source.moments(_metric(source, "observed"))) == 0
    assert populations == [4, 4, 4]


def test_nan_and_null_agree_across_backends() -> None:
    """The backend-divergence repro: one NaN y gave a y sum of 7.0 on
    pandas but NaN on polars/pyarrow. Both NaN and null are now one
    'missing' concept: identical refusal text by default, identical
    moments under missing='zero', on every installed backend."""
    frames: list[tuple[str, Any]] = [
        (
            "pyarrow-nan",
            pa.table(
                {
                    "unit": ["a", "b", "c", "d"],
                    "variant": ["control", "control", "treatment", "treatment"],
                    "y": [1.0, float("nan"), 3.0, 4.0],
                }
            ),
        )
    ]
    if importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        frames.append(
            (
                "pandas-nan",
                pd.DataFrame(
                    {
                        "unit": ["a", "b", "c", "d"],
                        "variant": ["control", "control", "treatment", "treatment"],
                        "y": [1.0, float("nan"), 3.0, 4.0],
                    }
                ),
            )
        )
    if importlib.util.find_spec("polars") is not None:
        import polars as pl

        frames.append(
            (
                "polars-nan",
                pl.DataFrame(
                    {
                        "unit": ["a", "b", "c", "d"],
                        "variant": ["control", "control", "treatment", "treatment"],
                        "y": [1.0, float("nan"), 3.0, 4.0],
                    }
                ),
            )
        )
        frames.append(
            (
                "polars-null",
                pl.DataFrame(
                    {
                        "unit": ["a", "b", "c", "d"],
                        "variant": ["control", "control", "treatment", "treatment"],
                        "y": [1.0, None, 3.0, 4.0],
                    }
                ),
            )
        )
    assert len(frames) >= 2

    codes: set[str] = set()
    moment_sets: list[list[tuple[str, int, float]]] = []
    for _, frame in frames:
        with pytest.raises(InvalidRequestError) as exc_info:
            from_unit_summary(
                frame, unit="unit", group="variant", control="control", metrics={"y": "mean"}
            )
        codes.add(exc_info.value.code)
        src = from_unit_summary(
            frame,
            unit="unit",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="y", missing="zero")],
        )
        moment_sets.append(
            sorted((str(r["group_id"]), r["n"], _sum_y(r)) for r in src.moments(_metric(src, "y")))
        )
    assert codes == {"frame.validation.metric_value_missing"}  # identical refusal on every backend
    assert all(m == moment_sets[0] for m in moment_sets)
    assert moment_sets[0] == [("control", 2, 1.0), ("treatment", 2, 7.0)]


def test_duplicate_metric_names_refused() -> None:
    """Duplicate names cross-pair arms downstream (previously measured: 4 estimates
    [0.1683, 0.1683, 1.3365, 1.3365] from one duplicated name)."""
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            _arrow_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                {"name": "m", "value_column": "revenue"},
                {"name": "m", "value_column": "orders"},
            ],
        )
    assert exc_info.value.code == "frame.duplicate_metric_name"
    assert exc_info.value.context["name"] == "m"


def _cuped_null_covariate_frame() -> pa.Table:
    """True lift 0, x strongly predictive of y, 50 covariate nulls in the
    control arm only - the shape that silently biased a true-zero lift to -7.2%."""
    rng = np.random.default_rng(7)
    n = 500
    units, variants, ys, xs = [], [], [], []
    for group in ("control", "treatment"):
        x = rng.normal(30.0, 2.0, n)
        y = 2.0 + 0.9 * x + rng.normal(0.0, 1.0, n)
        units += [f"{group}-{i}" for i in range(n)]
        variants += [group] * n
        ys += y.tolist()
        xs += x.tolist()
    x_col: list[float | None] = list(xs)
    for i in range(50):  # nulls in the CONTROL arm's covariate
        x_col[i] = None
    return pa.table({"user_id": units, "variant": variants, "y": ys, "x": x_col})


def test_cuped_covariate_nulls_auto_imputed_with_counted_warning() -> None:
    """Nulls in one arm's covariate deflated that arm's x-mean (sums skip
    nulls, n counts all rows), which theta turned into a confident bias:
    previously measured: lift -7.2%, CI [-11.3%, -2.9%] at TRUE LIFT ZERO.
    Pooled-mean imputation (deterministic, never using arm or outcome) is
    unbiased under randomization - the estimate must cover zero again,
    and the repair must be counted and surfaced, never silent."""
    with pytest.warns(IncrementWarning) as rec:
        src = from_unit_summary(
            _cuped_null_covariate_frame(),
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="y", covariate="x")],
            design=_DESIGN,
        )
    assert "frame.validation.metric_covariate_missing_imputed" in warning_codes(rec)
    (est,) = readouts.run(src, decision_method=Method(name="cuped", variance_reduction="cuped"))
    lift = est.require_lift()
    assert lift.lb is not None and lift.ub is not None
    assert lift.lb <= 0.0 <= lift.ub
    assert abs(lift.value) < 0.02


def test_all_null_covariate_refuses_with_actionable_message() -> None:
    """A covariate with NO observed values cannot yield a pooled mean; the
    refusal must name the column and the remedy on both the entry
    chokepoint and the moment-construction path (never a bare
    float(None) TypeError, which would also differ by backend)."""
    import narwhals as nw

    from increment.frame import _apply_spec_missing

    table = pa.table(
        {
            "group_id": ["control", "treatment"] * 4,
            "y": [1.0, 2.0] * 4,
            "x": [None] * 8,
        }
    )
    long = nw.from_native(table, eager_only=True).with_columns(
        nw.col("y").cast(nw.Float64), nw.col("x").cast(nw.Float64)
    )
    spec = MetricSpec(name="y", covariate="x")
    with pytest.raises(InvalidRequestError) as exc_info:
        _apply_spec_missing(long, spec, has_x=True, has_den=False)
    assert exc_info.value.code == "frame.validation.metric_covariate_impute_all_null"
    assert exc_info.value.context == {"metric": "y", "covariate": "x"}


def test_asof_source_without_first_exposure_anchor_refuses() -> None:
    import narwhals as nw

    from increment._frame_moments import _asof_source_with_indices

    panel = nw.from_native(pa.table({"unit_id": ["u1"]}), eager_only=True)
    with pytest.raises(InvalidRequestError) as exc_info:
        _asof_source_with_indices(panel, exposure=None, first_exposure=None, uptake="clicked")
    assert exc_info.value.code == "frame.moments.asof_source_uptake_needs_anchor"


def test_asof_unit_rows_completed_windows_unbounded_retention_refuses() -> None:
    """``_asof_unit_rows`` carries its own completed_windows_only guard,
    independent of any caller-side pre-check."""
    import narwhals as nw

    from increment._frame_moments import _asof_unit_rows
    from increment.frame import synthesise_metric

    spec = MetricSpec(name="d0_plus", type="retention", value_column="revenue", threshold_days=0)
    metric = synthesise_metric(spec)
    panel = nw.from_native(pa.table({"unit_id": ["u1"]}), eager_only=True)
    with pytest.raises(InvalidRequestError) as exc_info:
        _asof_unit_rows(
            panel,
            spec=spec,
            metric=metric,
            by=(),
            exposure=None,
            first_exposure=None,
            uptake=None,
            uptake_window_days=None,
            completed_windows_only=True,
        )
    assert exc_info.value.code == "frame.moments.asof_moments_completed"
    assert exc_info.value.context["metric"] == "d0_plus"


def test_asof_unit_rows_retention_without_exposure_refuses() -> None:
    """Retention needs an exposure-relative day index; without one,
    ``_asof_unit_rows`` itself refuses rather than computing a band."""
    from increment._frame_moments import _asof_unit_rows
    from increment.frame import synthesise_metric

    spec = MetricSpec(name="y", type="retention", value_column="revenue", threshold_days=(0, 3))
    metric = synthesise_metric(spec)
    panel = nw.from_native(pa.table({"unit_id": ["u1"], "ds": [date(2025, 1, 1)]}), eager_only=True)
    with pytest.raises(InvalidRequestError) as exc_info:
        _asof_unit_rows(
            panel,
            spec=spec,
            metric=metric,
            by=(),
            exposure=None,
            first_exposure=None,
            uptake=None,
            uptake_window_days=None,
            completed_windows_only=False,
        )
    assert exc_info.value.code == "frame.moments.asof_unit_rows"


def test_collapse_to_unit_totals_window_without_first_exposure_refuses() -> None:
    import narwhals as nw

    from increment._frame_moments import _collapse_to_unit_totals

    panel = nw.from_native(
        pa.table({"unit_id": ["u1"], "group_id": ["control"], "revenue": [1.0], "clicked": [1.0]}),
        eager_only=True,
    )
    spec = MetricSpec(name="revenue")
    with pytest.raises(InvalidRequestError) as exc_info:
        _collapse_to_unit_totals(
            panel, [spec], uptake="clicked", window_days=3, first_exposure=None
        )
    assert exc_info.value.code == "frame.moments.collapse_to_unit"


def test_covariate_imputation_computed_before_outcome_drop() -> None:
    """The covariate's pooled mean must be computed from the covariate
    column alone, before missing='drop' removes any outcome-null rows -
    regression: y=[1,None,3,4], x=[0,100,None,10] previously imputed 5.0
    (mean of the two x's surviving the y-drop) instead of the correct
    36.67 (mean of all three observed x's)."""
    table = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["control", "control", "treatment", "treatment"],
            "y": pa.array([1.0, None, 3.0, 4.0], type=pa.float64()),
            "x": pa.array([0.0, 100.0, None, 10.0], type=pa.float64()),
        }
    )
    with pytest.warns(UserWarning):
        src = from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="y", missing="drop", covariate="x")],
        )
    moments = {row["group_id"]: row for row in src.moments(_metric(src, "y"))}
    observed_x_mean = (0.0 + 100.0 + 10.0) / 3
    assert moments["control"]["n"] == 1  # the y=None unit is dropped
    assert moments["treatment"]["n"] == 2
    assert _sum_x(moments["control"]) == pytest.approx(0.0)
    assert _sum_x(moments["treatment"]) == pytest.approx(10.0 + observed_x_mean)


def test_cuped_covariate_missing_error_escape_hatch() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            _cuped_null_covariate_frame(),
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="y", covariate="x", covariate_missing="error")],
        )
    assert exc_info.value.code == "frame.validation.metric_covariate_column"
    assert exc_info.value.context["n_missing"] == 50
    assert exc_info.value.context["covariate"] == "x"


def test_enforce_missing_policy_all_null_impute_covariate_refuses() -> None:
    """A covariate with no observed values refuses by code through the
    public summary constructor."""
    table = pa.table(
        {
            "user_id": ["u1", "u2"],
            "variant": ["control", "treatment"],
            "y": pa.array([1.0, 2.0], type=pa.float64()),
            "x": pa.array([None, None], type=pa.float64()),
        }
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="y", covariate="x")],
        )
    assert exc_info.value.code == "frame.validation.metric_covariate_impute_all_null"
    assert exc_info.value.context["metric"] == "y"
    assert exc_info.value.context["covariate"] == "x"


def test_all_null_covariate_entry_points_share_canonical_refusal_code() -> None:
    """The moment reducer and public summary constructor expose one hazard."""
    import narwhals as nw

    from increment.frame import _apply_spec_missing

    long = nw.from_native(
        pa.table({"group_id": ["control", "treatment"], "y": [1.0, 2.0], "x": [None, None]}),
        eager_only=True,
    ).with_columns(nw.col("y").cast(nw.Float64), nw.col("x").cast(nw.Float64))
    spec = MetricSpec(name="y", covariate="x")
    with pytest.raises(InvalidRequestError) as via_moments:
        _apply_spec_missing(long, spec, has_x=True, has_den=False)

    with pytest.raises(InvalidRequestError) as via_validation:
        from_unit_summary(
            pa.table(
                {
                    "user_id": ["u1", "u2"],
                    "variant": ["control", "treatment"],
                    "y": [1.0, 2.0],
                    "x": pa.array([None, None], type=pa.float64()),
                }
            ),
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="y", covariate="x")],
        )

    assert (
        via_moments.value.code
        == via_validation.value.code
        == "frame.validation.metric_covariate_impute_all_null"
    )


def test_cuped_covariate_missing_zero_uses_dense_zero_convention() -> None:
    """covariate_missing='zero' mirrors the definitions path's dense x=0
    imputation: sum(x) is exactly the observed sum, n still counts every
    row, and the moments carry no NaN on any backend."""
    table = _cuped_null_covariate_frame()
    with pytest.warns(IncrementWarning) as rec:
        src = from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="y", covariate="x", covariate_missing="zero")],
        )
    assert "frame.validation.metric_covariate_missing_imputed" in warning_codes(rec)
    moments = {r["group_id"]: r for r in src.moments(_metric(src, "y"))}
    observed_sum = sum(v for v in table["x"].to_pylist()[:500] if v is not None)
    assert moments["control"]["n"] == 500
    assert _sum_x(moments["control"]) == pytest.approx(observed_sum)
    assert math.isfinite(moments["control"]["cx2"])


def test_covariate_missing_without_covariate_is_refused() -> None:
    with pytest.raises(InvalidRequestError) as error:
        MetricSpec(name="y", covariate_missing="error")
    assert error.value.code == "frame.metric.sets_covariate_missing"


def test_panel_null_metric_value_refuses_and_zero_fills() -> None:
    base = date(2025, 1, 1)
    table = pa.table(
        {
            "user_id": [u for u in ("a", "b", "c", "d") for _ in range(2)],
            "variant": [
                g for g in ("control", "control", "treatment", "treatment") for _ in range(2)
            ],
            "day": [base + timedelta(days=k) for _ in range(4) for k in range(2)],
            "revenue": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    kwargs: dict[str, Any] = {
        "unit": "user_id",
        "group": "variant",
        "date": "day",
        "control": "control",
    }
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_panel(table, metrics={"revenue": "mean"}, **kwargs)
    assert exc_info.value.code == "frame.validation.metric_value_missing"
    assert exc_info.value.context["metric"] == "revenue"
    assert exc_info.value.context["column"] == "revenue"
    assert exc_info.value.context["n_missing"] == 1
    src = from_unit_panel(table, metrics=[MetricSpec(name="revenue", missing="zero")], **kwargs)
    out = {r["group_id"]: r for r in src.moments(_metric(src, "revenue"), grain="total")}
    assert _sum_y(out["control"]) == pytest.approx(7.0)  # 1+2 + (0+4)
    assert out["control"]["n"] == 2


def test_panel_missing_drop_refused_as_indistinguishable_from_zero() -> None:
    base = date(2025, 1, 1)
    table = pa.table(
        {
            "user_id": ["a", "b"],
            "variant": ["control", "treatment"],
            "day": [base, base],
            "revenue": [1.0, 2.0],
        }
    )
    with pytest.raises(UnsupportedRequestError) as exc:
        from_unit_panel(
            table,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[MetricSpec(name="revenue", missing="drop")],
        )
    assert exc.value.code == "frame.missing_policy.panel_drop"
    assert exc.value.context["missing"] == "drop"


def test_unit_frame_honours_declared_missing_policy() -> None:
    """unit_frame must serve the same declared semantics as the moments:
    zero-filled values under missing='zero', complete-case rows under
    missing='drop' - never a raw null the chokepoint already ruled on."""
    table = _half_null_table()
    src_zero = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue", missing="zero")],
    )
    uf = nw.from_native(src_zero.unit_frame(_metric(src_zero, "revenue")), eager_only=True)
    assert uf.shape[0] == 200
    assert float(uf["y"].sum()) == pytest.approx(
        sum(v or 0.0 for v in table["revenue"].to_pylist())
    )
    with pytest.warns(IncrementWarning) as rec:
        src_drop = from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="revenue", missing="drop")],
        )
    assert "frame.validation.metric_missing_drop" in warning_codes(rec)
    uf_drop = nw.from_native(src_drop.unit_frame(_metric(src_drop, "revenue")), eager_only=True)
    assert uf_drop.shape[0] == 150


def test_unit_frame_ratio_missing_policy_covers_the_denominator_too() -> None:
    """A ratio metric's missing='zero'/'drop' apply to y_den exactly as to
    y (mirroring _apply_spec_missing's combined predicate) - a null
    denominator is not a hole the numerator's policy alone papers over."""
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(6)],
            "variant": ["control", "control", "control", "treatment", "treatment", "treatment"],
            "revenue": [10.0, None, 30.0, 40.0, 50.0, 60.0],
            "orders": [2.0, 3.0, None, 4.0, 5.0, 6.0],
        }
    )
    src_zero = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(
                name="aov",
                type="ratio",
                numerator="revenue",
                denominator="orders",
                missing="zero",
            )
        ],
    )
    uf_zero = nw.from_native(src_zero.unit_frame(_metric(src_zero, "aov")), eager_only=True)
    assert uf_zero.shape[0] == 6
    assert uf_zero["y"].is_null().sum() == 0
    assert float(uf_zero["y"].sum()) == 190.0
    assert uf_zero["y_den"].is_null().sum() == 0
    assert float(uf_zero["y_den"].sum()) == 20.0

    with pytest.warns(IncrementWarning) as rec:
        src_drop = from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(
                    name="aov",
                    type="ratio",
                    numerator="revenue",
                    denominator="orders",
                    missing="drop",
                )
            ],
        )
    assert "frame.validation.metric_missing_drop" in warning_codes(rec)
    uf_drop = nw.from_native(src_drop.unit_frame(_metric(src_drop, "aov")), eager_only=True)
    # 2 of 6 rows drop: one null revenue, one null orders (distinct rows).
    assert uf_drop.shape[0] == 4


def test_terse_quantile_redirects_to_explicit_metricspec() -> None:
    """The terse form cannot carry quantile= - the message must point at
    the explicit MetricSpec form, exactly as the retention redirect does,
    instead of claiming quantile is an unknown type."""
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            _arrow_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "quantile"},
        )
    assert exc_info.value.code == "frame.metric_type_quantile"
    assert exc_info.value.context["name"] == "revenue"


def test_control_not_found_on_empty_frame_names_emptiness() -> None:
    table = pa.table(
        {
            "user_id": pa.array([], type=pa.string()),
            "variant": pa.array([], type=pa.string()),
            "revenue": pa.array([], type=pa.float64()),
        }
    )
    with pytest.raises(InvalidRequestError) as error:
        from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
        )
    assert error.value.code == "source.frame.control_missing"


def test_summary_conversion_metric_must_be_binary(arrow_frame: pa.Table) -> None:
    """type='conversion' is estimator-identical to mean, so a magnitude
    column wearing the label ships a number whose NAME promises a rate.
    Refused at the chokepoint, naming type='mean' as the honest choice."""
    with pytest.raises(InvalidRequestError) as error:
        from_unit_summary(
            arrow_frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="revenue", type="conversion")],
        )
    assert error.value.code == "source.frame.conversion_not_binary"


def test_summary_conversion_accepts_bool_and_declared_zero_nulls() -> None:
    """True/False columns and nulls under a declared missing='zero' stay
    accepted - the binary check rules on observed values only."""
    table = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["control", "control", "treatment", "treatment"],
            "converted": [True, False, True, None],
        }
    )
    src = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="converted", type="conversion", missing="zero")],
    )
    out = {r["group_id"]: r for r in src.moments(_metric(src, "converted"), grain="total")}
    assert _sum_y(out["treatment"]) == pytest.approx(1.0)


def test_tz_aware_date_column_refused_with_day_grain_contract() -> None:
    """A tz-aware day column used to crash with a raw backend cast
    TypeError naming nothing; the refusal must name the column and the
    caller-owns-bucketing contract."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "variant": ["control", "control", "treatment", "treatment"],
            "day": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"]
            ).tz_localize("US/Eastern"),
            "revenue": [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.raises(InvalidRequestError) as error:
        from_unit_panel(
            df,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
        )
    assert error.value.code == "source.panel.day_grain_columns"


def test_tz_aware_exposure_date_column_refused() -> None:
    pd = pytest.importorskip("pandas")
    naive_days = pd.to_datetime(["2025-01-01", "2025-01-02"] * 2)
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "variant": ["control", "control", "treatment", "treatment"],
            "day": naive_days,
            "exposed_on": pd.to_datetime(["2025-01-01"] * 4).tz_localize("UTC"),
            "revenue": [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.raises(InvalidRequestError) as error:
        from_unit_panel(
            df,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[MetricSpec(name="revenue", window_days=2)],
            exposure_date="exposed_on",
        )
    assert error.value.code == "source.panel.day_grain_columns"


def test_null_unit_id_refused_on_summary() -> None:
    """A null unit id would fail the densification/fold-hash join on some
    backends and be silently manufactured or dropped on others - refused
    at the boundary, before any duplicate check runs."""
    table = pa.table(
        {
            "user_id": ["u1", None],
            "variant": ["control", "treatment"],
            "revenue": [1.0, 2.0],
        }
    )
    with pytest.raises(InvalidRequestError) as error:
        from_unit_summary(
            table, unit="user_id", group="variant", control="control", metrics={"revenue": "mean"}
        )
    assert error.value.code == "source.frame.null_identity"


def test_null_unit_id_refused_on_panel() -> None:
    base = date(2025, 1, 1)
    table = pa.table(
        {
            "user_id": ["u1", None],
            "variant": ["control", "treatment"],
            "day": [base, base],
            "revenue": [1.0, 2.0],
        }
    )
    with pytest.raises(InvalidRequestError) as error:
        from_unit_panel(
            table,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
        )
    assert error.value.code == "source.frame.null_identity"


def test_null_date_refused_on_panel() -> None:
    """A null day axis value would silently zero-fill on pandas, raise a
    bare backend TypeError on polars, or a bare ArrowInvalid on pyarrow -
    refused at the boundary instead, naming the date column."""
    table = pa.table(
        {
            "user_id": ["u1", "u1"],
            "variant": ["control", "control"],
            "day": pa.array([date(2025, 1, 1), None]),
            "revenue": [1.0, 2.0],
        }
    )
    with pytest.raises(InvalidRequestError) as error:
        from_unit_panel(
            table,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
        )
    assert error.value.code == "source.frame.null_identity"


def test_infinite_metric_value_refused_on_summary() -> None:
    """+-inf is neither null nor NaN, so it survives every missing-value
    check and then diverges by backend at the first sum (inf on pandas,
    NaN on polars/pyarrow) - refused up front instead, naming the
    offending column."""
    table = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["control", "control", "treatment", "treatment"],
            "revenue": [1.0, float("inf"), 3.0, 4.0],
        }
    )
    with pytest.raises(InvalidRequestError) as error:
        from_unit_summary(
            table, unit="user_id", group="variant", control="control", metrics={"revenue": "mean"}
        )
    assert error.value.code == "source.frame.non_finite"


def test_infinite_denominator_value_refused_on_summary() -> None:
    table = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["control", "control", "treatment", "treatment"],
            "revenue": [1.0, 2.0, 3.0, 4.0],
            "orders": [1.0, float("-inf"), 1.0, 2.0],
        }
    )
    with pytest.raises(InvalidRequestError) as error:
        from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(name="aov", type="ratio", numerator="revenue", denominator="orders")
            ],
        )
    assert error.value.code == "source.frame.non_finite"


def _null_exposure_panel() -> tuple[pa.Table, dict[str, Any]]:
    """5 units, u5 (treatment) with a null exposure date on every row;
    previously it silently vanished from windowed moments (n=4) while
    unit_counts()/srm() still counted it (n=5)."""
    base = date(2025, 1, 1)
    rows: list[tuple[str, str, date, date | None, float]] = []
    units = [
        ("u1", "control", base),
        ("u2", "control", base),
        ("u3", "treatment", base),
        ("u4", "treatment", base),
        ("u5", "treatment", None),
    ]
    for u, g, exp in units:
        for k in range(3):
            rows.append((u, g, base + timedelta(days=k), exp, float(k + 1)))
    table = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "exposed_on", "revenue"],
                list(zip(*rows, strict=True)),
                strict=True,
            )
        )
    )
    kwargs: dict[str, Any] = {
        "unit": "user_id",
        "group": "variant",
        "date": "day",
        "control": "control",
        "metrics": [MetricSpec(name="revenue", window_days=2)],
        "exposure_date": "exposed_on",
        "observation_end": date(2025, 1, 10),
    }
    return table, kwargs


def test_null_exposure_date_refuses_naming_the_exclude_knob() -> None:
    table, kwargs = _null_exposure_panel()
    with pytest.raises(InvalidRequestError) as error:
        from_unit_panel(table, **kwargs)
    assert error.value.code == "source.panel.null_exposure"


def test_null_exposure_date_excluded_with_accounting() -> None:
    """on_unassigned='exclude': the anchorless unit leaves the moments AND
    the unit counts together, surfaced once under '(unassigned)' - one
    unusable-unit concept, consistent between moments and the SRM check."""
    table, kwargs = _null_exposure_panel()
    src = from_unit_panel(table, **kwargs, on_unassigned="exclude")
    assert src.unit_counts() == {"control": 2, "treatment": 2, "(unassigned)": 1}
    out = {r["group_id"]: r for r in src.moments(_metric(src, "revenue"), grain="total")}
    assert out["treatment"]["n"] == 2
    assert out["control"]["n"] == 2
    # window [0, 2) keeps days 1 and 2 of each unit: 1.0 + 2.0 per unit
    assert _sum_y(out["treatment"]) == pytest.approx(6.0)


def test_summary_null_exposure_excluded_with_accounting() -> None:
    """The unit summary counts a null-exposure exclusion like the panel does."""
    table = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5"],
            "variant": ["control", "control", "treatment", "treatment", "treatment"],
            "exposed_on": [date(2025, 1, 1)] * 4 + [None],
            "revenue": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    src = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        exposure_date="exposed_on",
        on_unassigned="exclude",
    )
    assert src.unit_counts() == {"control": 2, "treatment": 2, "(unassigned)": 1}


def test_densified_cells_counts_declared_zero_null_fills() -> None:
    """A genuine in-row null under missing='zero' is manufactured into a
    zero by densification exactly like an absent (unit, day) cell - the
    counter must include both, not just the synthesised spine rows."""
    table = pa.table(
        {
            "user_id": ["u1", "u1", "u2"],
            "variant": ["control", "control", "treatment"],
            "day": ["d1", "d2", "d1"],
            "revenue": [1.0, None, 2.0],
        }
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", missing="zero")],
    )
    # 1 synthesised (u2, d2) cell + 1 genuine-null (u1, d2) cell
    assert src.densified_cells == 2


def test_single_unit_arm_error_names_metric_and_arm() -> None:
    """A stray 1-unit arm still refuses at estimation time (a moment-grain
    read is legitimate), but the error must name the metric and the arm
    instead of a bare ddof=1 complaint."""
    rows = [*_ROWS, ("u13", "stray", 5.0, 1.0, 1.0, 1.0, 1.0)]
    cols = list(zip(*rows, strict=True))
    table = pa.table(dict(zip(_COLUMNS, cols, strict=True)))
    src = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        design=_DESIGN,
    )
    with pytest.raises(InvalidRequestError) as error:
        readouts.run(src)
    assert error.value.code == "estimation.armstats.arm_stats.least_compute_metric"


def test_windowed_encouragement_panel_offers_asof_but_refuses_total() -> None:
    base = date(2025, 1, 1)
    rows = []
    for unit, variant, revenue in (
        ("c1", "control", 1.0),
        ("c2", "control", 3.0),
        ("t1", "treatment", 2.0),
        ("t2", "treatment", 6.0),
    ):
        for offset in range(5):
            rows.append(
                {
                    "user_id": unit,
                    "variant": variant,
                    "day": base + timedelta(days=offset),
                    "exposed_on": base,
                    "clicked": float(variant == "treatment" and offset < 1),
                    "revenue": revenue if offset < 2 else 1000.0,
                }
            )
    src = from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", window_days=2)],
        uptake="clicked",
        design=_encouragement_design(window_days=3),
        exposure_date="exposed_on",
    )
    metric = src.context.metrics[0]

    assert src.moments(metric, grain="asof")
    with pytest.raises(CapabilityError) as exc_grain:
        src.moments(metric, grain="total")
    assert exc_grain.value.code == "source.frame.windowed_encouragement_total"


def test_encouragement_retention_remains_refused_at_construction() -> None:
    base = date(2025, 1, 1)
    table = pa.table(
        {
            "user_id": ["c1", "c1", "t1", "t1"],
            "variant": ["control", "control", "treatment", "treatment"],
            "day": [base, base + timedelta(days=1)] * 2,
            "exposed_on": [base] * 4,
            "clicked": [0.0, 0.0, 1.0, 0.0],
            "returned": [0.0, 1.0, 0.0, 1.0],
        }
    )
    with pytest.raises(CapabilityError) as exc:
        from_unit_panel(
            table,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[MetricSpec(name="returned", type="retention", threshold_days=(1, 3))],
            uptake="clicked",
            design=_encouragement_design(window_days=3),
            exposure_date="exposed_on",
        )
    assert exc.value.code == "readout.encouragement.retention"


def test_encouragement_retention_invalid_band_still_refuses_encouragement_not_validation() -> None:
    """A structurally invalid band must not let synthesis run first and mask the
    real refusal behind a raw pydantic ValidationError (regression: the capability
    check must fire before metric synthesis on the panel path)."""
    base = date(2025, 1, 1)
    table = pa.table(
        {
            "user_id": ["c1", "c1", "t1", "t1"],
            "variant": ["control", "control", "treatment", "treatment"],
            "day": [base, base + timedelta(days=1)] * 2,
            "exposed_on": [base] * 4,
            "clicked": [0.0, 0.0, 1.0, 0.0],
            "returned": [0.0, 1.0, 0.0, 1.0],
        }
    )
    with pytest.raises(CapabilityError) as exc:
        from_unit_panel(
            table,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[MetricSpec(name="returned", type="retention", threshold_days=(5, 3))],
            uptake="clicked",
            design=_encouragement_design(window_days=3),
            exposure_date="exposed_on",
        )
    assert exc.value.code == "readout.encouragement.retention"


def _windowed_encouragement_source(
    *, outcome_window_days: int, uptake_window_days: int, periods: int
):
    base = date(2025, 1, 1)
    rows = []
    for unit, variant, value, click_offset in (
        ("c1", "control", 1.0, None),
        ("c2", "control", 3.0, None),
        ("t1", "treatment", 2.0, 2),
        ("t2", "treatment", 6.0, 0),
    ):
        for offset in range(periods):
            rows.append(
                {
                    "user_id": unit,
                    "variant": variant,
                    "day": base + timedelta(days=offset),
                    "exposed_on": base,
                    "clicked": float(offset == click_offset),
                    "revenue": value if offset < outcome_window_days else 1000.0,
                }
            )
    src = from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", window_days=outcome_window_days)],
        uptake="clicked",
        design=_encouragement_design(window_days=uptake_window_days),
        exposure_date="exposed_on",
    )
    return src, src.context.metrics[0], base


def test_asof_encouragement_completed_windows_gate_later_edge() -> None:
    src, metric, base = _windowed_encouragement_source(
        outcome_window_days=2,
        uptake_window_days=3,
        periods=5,
    )

    rows = src.moments(metric, grain="asof", completed_windows_only=True)

    assert min(row["ds"] for row in rows) == base + timedelta(days=3)
    treatment = [row for row in rows if row["group_id"] == "treatment"]
    assert {row["sum_d"] for row in treatment} == {2.0}
    assert {row["ref_y"] for row in treatment} == {8.0}


def test_censoring_warning_attributed_to_caller_not_engine_internals() -> None:
    """The warning's source line must survive changes to the internal call
    chain: it is attributed to the first frame outside the package, not a
    hardcoded number of frames up."""
    src = _fence_panel(date(2025, 1, 13))
    metric = _metric(src, "revenue")
    with pytest.warns(IncrementWarning) as record:
        src.moments(metric, grain="total")
    (warning,) = [
        r
        for r in record
        if isinstance(r.message, IncrementWarning)
        and r.message.code == "frame.censoring.dropped_units"
    ]
    assert warning.filename == __file__


def test_missing_impute_is_refused_by_every_frame_constructor_with_one_code() -> None:
    spec = {"name": "revenue", "missing": "impute"}
    with pytest.raises(InvalidRequestError) as summary:
        from_unit_summary(
            _arrow_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics=[spec],
        )
    with pytest.raises(InvalidRequestError) as panel:
        from_unit_panel(
            _panel_table(),
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[spec],
        )
    for raised in (summary, panel):
        assert raised.value.code == "frame.metric.missing_impute"
        assert raised.value.context["metric"] == "revenue"
