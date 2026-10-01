"""to_frame serialization of non-str Sequence cells."""

from __future__ import annotations

import math
import warnings
from typing import Any, cast

import pandas as pd
import pytest

from increment.breakout.estimates import run_breakout
from tests.breakout.test_estimates import _make_arm_row, _mean_metric


def _bh_rows():
    return [
        _make_arm_row(100, 10.0, 4.0, country="US", metric="rev", group_id="control"),
        _make_arm_row(100, 8.0, 4.0, country="US", metric="rev", group_id="treatment"),
        _make_arm_row(100, 10.0, 4.0, country="CA", metric="rev", group_id="control"),
        _make_arm_row(100, 11.0, 4.0, country="CA", metric="rev", group_id="treatment"),
    ]


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_to_frame_serializes_family_axes_tuple_on_every_backend(backend):
    pytest.importorskip(backend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = run_breakout(
            _bh_rows(),
            [_mean_metric("rev")],
            control_group="control",
            dimension="country",
            correction="bh",
        )
    frame: Any = out.to_frame(backend=backend)
    if backend == "pandas":
        values = frame["family_axes"].tolist()
    elif backend == "polars":
        values = frame["family_axes"].to_list()
    else:
        values = frame.column("family_axes").to_pylist()
    assert values[0] == "('metric', 'arm', 'segment')"


def _directional_bh_rows():
    return [
        _make_arm_row(500, 10.0, 4.0, country="US", metric="rev", group_id="control"),
        _make_arm_row(500, 12.0, 4.0, country="US", metric="rev", group_id="treatment"),
        _make_arm_row(500, 10.0, 4.0, country="CA", metric="rev", group_id="control"),
        _make_arm_row(500, 11.5, 4.0, country="CA", metric="rev", group_id="treatment"),
    ]


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_to_frame_open_side_column_on_directional_bh_breakout(backend):
    """``correction="bh"`` + ``alternative="greater"`` opens the far bound
    on every selected cell -- the flattened frame must carry an explicit
    ``open_side`` column so a consumer can tell a genuinely unbounded
    endpoint from a closed one without inspecting ``lb``/``ub`` nulls
    alone, on every supported backend."""
    pytest.importorskip(backend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = run_breakout(
            _directional_bh_rows(),
            [_mean_metric("rev")],
            control_group="control",
            dimension="country",
            correction="bh",
            alternative="greater",
            q=0.5,
        )
    frame: Any = out.to_frame(backend=backend)
    if backend == "pandas":
        rows = frame[["dimension_value", "lift", "lb", "ub", "open_side"]].to_dict("records")
    elif backend == "polars":
        rows = frame.select(["dimension_value", "lift", "lb", "ub", "open_side"]).to_dicts()
    else:
        rows = frame.select(["dimension_value", "lift", "lb", "ub", "open_side"]).to_pylist()
    by_segment = {r["dimension_value"]: r for r in rows}
    assert set(by_segment) == {"US", "CA"}
    for segment in ("US", "CA"):
        row = by_segment[segment]
        assert row["lift"] is not None
        assert row["lb"] is not None
        assert row["ub"] is None or (isinstance(row["ub"], float) and math.isnan(row["ub"]))
        assert row["open_side"] == "upper"


def test_to_frame_open_side_distinguishes_open_from_closed_and_unavailable():
    """Direct construction (not run_breakout) pinning the three states
    ``open_side`` must tell apart: a closed interval (both bounds finite,
    ``open_side=None``), a genuinely unbounded one-sided interval
    (``open_side`` set, exactly one bound ``None``), and an unavailable
    row (``lift=None``, every flattened column including ``open_side``
    is null)."""
    from increment.breakout.estimates import BreakoutEstimate, BreakoutEstimates
    from increment.estimation.results import Estimate

    closed = BreakoutEstimate(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        dimension="country",
        dimension_value="US",
        source=None,
        lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
    )
    unavailable = BreakoutEstimate(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        dimension="country",
        dimension_value="CA",
        source=None,
        lift=None,
        excluded="zero_variance",
    )
    open_row = BreakoutEstimate(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        dimension="country",
        dimension_value="MX",
        source=None,
        lift=Estimate(value=0.1, lb=0.05, ub=None, level=0.95, alpha=0.05, open_side="upper"),
    )
    frame = cast(
        pd.DataFrame, BreakoutEstimates([closed, unavailable, open_row]).to_frame(backend="pandas")
    )
    by_segment = frame.set_index("dimension_value")
    assert by_segment.loc["US", "lb"] == pytest.approx(0.05)
    assert by_segment.loc["US", "ub"] == pytest.approx(0.15)
    assert pd.isna(by_segment.loc["US", "open_side"])

    assert pd.isna(by_segment.loc["CA", "lift"])
    assert pd.isna(by_segment.loc["CA", "lb"])
    assert pd.isna(by_segment.loc["CA", "ub"])
    assert pd.isna(by_segment.loc["CA", "open_side"])

    assert by_segment.loc["MX", "lb"] == pytest.approx(0.05)
    assert pd.isna(by_segment.loc["MX", "ub"])
    assert by_segment.loc["MX", "open_side"] == "upper"
