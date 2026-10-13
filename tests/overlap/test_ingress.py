"""Ingress validation and parity: pandas, polars, pyarrow, and DuckDB through Ibis.

Every hazard raises the same refusal code on every ingress before any moment is
computed, and identical per-unit data yields identical emitted rows.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable
from typing import Any

import ibis
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from increment.errors import CodedError
from increment.overlap import (
    CONTRASTS,
    ExperimentColumns,
    OverlapCells,
    estimate_overlap_effects,
)
from tests.overlap.fixtures import (
    COLUMNS,
    DESIGN,
    from_frame,
    from_table,
    generated_cells,
    unit_frame,
)

pytestmark = pytest.mark.integration

MU = {(0, 0): 1.0, (1, 0): 1.5, (0, 1): 1.2, (1, 1): 2.4}
SIZES = {(0, 0): 40, (1, 0): 40, (0, 1): 15, (1, 1): 12}


def base_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return unit_frame(
        generated_cells(rng, MU, SIZES), first_only=[3.0] * 5, second_only=[4.0] * 4, neither=[0.0]
    )


def _pandas(frame: pd.DataFrame, **kw: Any) -> OverlapCells:
    return from_frame(frame, **kw)


def _polars(frame: pd.DataFrame, **kw: Any) -> OverlapCells:
    return from_frame(pl.from_pandas(frame), **kw)


def _pyarrow(frame: pd.DataFrame, **kw: Any) -> OverlapCells:
    return from_frame(pa.Table.from_pandas(frame, preserve_index=False), **kw)


def _duckdb(frame: pd.DataFrame, **kw: Any) -> OverlapCells:
    con = ibis.duckdb.connect()
    return from_table(con.create_table("units", frame), **kw)


INGRESSES: dict[str, Callable[..., OverlapCells]] = {
    "pandas": _pandas,
    "polars": _polars,
    "pyarrow": _pyarrow,
    "duckdb": _duckdb,
}


@pytest.mark.parametrize("ingress", sorted(INGRESSES))
def test_every_ingress_emits_the_pandas_rows(ingress: str) -> None:
    frame = base_frame()
    reference = estimate_overlap_effects(
        _pandas(frame), contrasts=CONTRASTS, roles={"revenue": "primary"}
    )
    readout = estimate_overlap_effects(
        INGRESSES[ingress](frame), contrasts=CONTRASTS, roles={"revenue": "primary"}
    )
    assert readout.audit == reference.audit
    assert readout.outcome_window == reference.outcome_window == dt.timedelta(days=14)
    assert readout.assignment_check == reference.assignment_check
    for got, want in zip(readout.rows, reference.rows, strict=True):
        for field in ("estimate", "se", "df", "lb", "ub", "p_value", "adjusted_p_value"):
            assert getattr(got, field) == pytest.approx(getattr(want, field), rel=1e-9, abs=1e-12)
        assert got.model_dump(
            exclude={"estimate", "se", "df", "lb", "ub", "p_value", "adjusted_p_value"}
        ) == want.model_dump(
            exclude={"estimate", "se", "df", "lb", "ub", "p_value", "adjusted_p_value"}
        )


def _set(frame: pd.DataFrame, row: int, column: str, value: object) -> pd.DataFrame:
    out = frame.copy()
    if value is None and out[column].dtype == bool:
        out[column] = out[column].astype(object)
    out.loc[row, column] = value
    return out


def _first_target_row(frame: pd.DataFrame) -> int:
    return int(np.flatnonzero(frame["elig_a"].astype(bool) & frame["elig_b"].astype(bool))[0])


def _make_late_start(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.loc[0, "start"] = out.loc[0, "at_b"] - dt.timedelta(minutes=1)
    return out


def _unequal_windows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.loc[0, "end"] = out.loc[0, "end"] + dt.timedelta(days=1)
    return out


def _aware_windows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("start", "end"):
        out[column] = out[column].dt.tz_localize("UTC")
    return out


def _empty_target(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["elig_b"] = False
    out["arm_b"] = pd.Series([None] * len(out), dtype="string")
    out["at_b"] = pd.NaT
    return out


HAZARDS: dict[str, tuple[Callable[[pd.DataFrame], pd.DataFrame], dict[str, object]]] = {
    "overlap.ingress.missing_columns": (lambda f: f.drop(columns=["at_b"]), {}),
    "overlap.ingress.columns_unbound": (
        lambda f: f,
        {"columns": {"checkout": COLUMNS["checkout"]}},
    ),
    "overlap.ingress.no_metrics": (lambda f: f, {"metrics": {}}),
    "overlap.ingress.unit_missing": (lambda f: _set(f, 3, "unit_id", None), {}),
    "overlap.ingress.duplicate_units": (lambda f: _set(f, 3, "unit_id", f.loc[4, "unit_id"]), {}),
    "overlap.ingress.eligibility_missing": (lambda f: _set(f, 3, "elig_a", None), {}),
    "overlap.ingress.ineligible_assigned": (lambda f: _set(f, len(f) - 1, "arm_a", "control"), {}),
    "overlap.ingress.eligible_unassigned": (lambda f: _set(f, 3, "arm_a", None), {}),
    "overlap.ingress.unknown_arm": (lambda f: _set(f, 3, "arm_b", "beta"), {}),
    "overlap.ingress.assignment_time_missing": (lambda f: _set(f, 3, "at_a", pd.NaT), {}),
    "overlap.target.empty": (_empty_target, {}),
    "overlap.timing.window_missing": (lambda f: _set(f, 0, "end", pd.NaT), {}),
    "overlap.timing.window_order": (lambda f: _set(f, 0, "end", f.loc[0, "start"]), {}),
    "overlap.timing.outcome_precedes_assignment": (_make_late_start, {}),
    "overlap.timing.horizon_unaligned": (_unequal_windows, {}),
    "overlap.timing.dtype": (lambda f: f.assign(start=f["start"].astype(str)), {}),
    "overlap.timing.awareness_mixed": (_aware_windows, {}),
    "overlap.outcome.missing": (lambda f: _set(f, 0, "revenue", None), {}),
    "overlap.outcome.nonfinite": (lambda f: _set(f, 0, "revenue", math.inf), {}),
    "overlap.outcome.non_numeric": (lambda f: f.assign(revenue=f["revenue"].astype(str)), {}),
}


@pytest.mark.parametrize("ingress", sorted(INGRESSES))
@pytest.mark.parametrize("code", sorted(HAZARDS))
def test_each_hazard_raises_one_code_on_every_ingress(code: str, ingress: str) -> None:
    mutate, overrides = HAZARDS[code]
    with pytest.raises(CodedError) as exc:
        INGRESSES[ingress](mutate(base_frame()), **overrides)
    assert exc.value.code == code


def test_out_of_target_outcomes_and_timing_are_never_read() -> None:
    frame = base_frame()
    outside = ~(frame["elig_a"] & frame["elig_b"])
    dirty = frame.copy()
    dirty.loc[outside, "revenue"] = np.nan
    dirty.loc[outside, "start"] = pd.NaT
    for ingress in INGRESSES.values():
        clean = estimate_overlap_effects(ingress(frame), contrasts=("joint_launch",))
        assert estimate_overlap_effects(ingress(dirty), contrasts=("joint_launch",)) == clean


def test_integer_eligibility_flags_are_read_as_booleans() -> None:
    frame = base_frame()
    flags = frame.assign(elig_a=frame["elig_a"].astype(int), elig_b=frame["elig_b"].astype(int))
    for ingress in INGRESSES.values():
        assert ingress(flags).audit == ingress(frame).audit


def test_assignment_time_after_the_window_opens_is_refused_by_column_role() -> None:
    frame = base_frame()
    late = frame.copy()
    row = _first_target_row(frame)
    late.loc[row, "at_a"] = late.loc[row, "start"] + dt.timedelta(seconds=1)
    for ingress in INGRESSES.values():
        with pytest.raises(CodedError) as exc:
            ingress(late)
        assert exc.value.code == "overlap.timing.outcome_precedes_assignment"


def test_columns_must_name_declared_experiments() -> None:
    other = {
        "checkout": COLUMNS["checkout"],
        "pricing": ExperimentColumns(eligible="elig_b", assignment="arm_b", assigned_at="at_b"),
    }
    with pytest.raises(CodedError) as exc:
        _pandas(base_frame(), columns=other)
    assert exc.value.code == "overlap.ingress.columns_unbound"


def test_cells_cannot_be_constructed_outside_the_ingresses() -> None:
    with pytest.raises(TypeError):
        OverlapCells(
            design=DESIGN,
            metrics=(),
            moments=(),
            audit=None,  # ty: ignore[invalid-argument-type] -- construction is refused before use
            outcome_window=None,  # ty: ignore[invalid-argument-type] -- construction is refused before use
        )
