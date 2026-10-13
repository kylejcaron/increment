"""Per-unit fixtures for the overlapping-experiments family and an independent reference.

``unit_frame`` lays out explicit outcomes per joint cell plus units eligible for
one experiment or neither, so every cell moment is known by construction.
``hc2_reference`` recomputes a contrast from the per-unit rows by ordinary
least squares on the saturated design ``[1, A, B, AB]`` with an HC2 sandwich,
sharing no code with the cell-means estimator.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from increment.overlap import ExperimentColumns, OverlapCells, OverlapDesign, OverlapExperiment

T0 = dt.datetime(2026, 3, 2, 9, 0)
START = T0 + dt.timedelta(days=1)
END = START + dt.timedelta(days=14)

FIRST = OverlapExperiment(name="checkout", treatment_probability=0.5)
SECOND = OverlapExperiment(
    name="search", control="old", treatment="new", treatment_probability=0.25
)
DESIGN = OverlapDesign(first=FIRST, second=SECOND)
COLUMNS = {
    "checkout": ExperimentColumns(eligible="elig_a", assignment="arm_a", assigned_at="at_a"),
    "search": ExperimentColumns(eligible="elig_b", assignment="arm_b", assigned_at="at_b"),
}
METRICS = {"revenue": "revenue"}


def from_frame(
    frame: Any,
    design: OverlapDesign = DESIGN,
    *,
    columns: Mapping[str, ExperimentColumns] = COLUMNS,
    metrics: Mapping[str, str] = METRICS,
) -> OverlapCells:
    """The fixture's column bindings through ``OverlapCells.from_frame``."""
    return OverlapCells.from_frame(
        frame,
        design,
        unit="unit_id",
        columns=columns,
        outcome_start="start",
        outcome_end="end",
        metrics=metrics,
    )


def from_table(
    table: Any,
    design: OverlapDesign = DESIGN,
    *,
    columns: Mapping[str, ExperimentColumns] = COLUMNS,
    metrics: Mapping[str, str] = METRICS,
) -> OverlapCells:
    """The fixture's column bindings through ``OverlapCells.from_table``."""
    return OverlapCells.from_table(
        table,
        design,
        unit="unit_id",
        columns=columns,
        outcome_start="start",
        outcome_end="end",
        metrics=metrics,
    )


def unit_frame(
    cells: Mapping[tuple[int, int], Sequence[float]],
    *,
    first_only: Sequence[float] = (),
    second_only: Sequence[float] = (),
    neither: Sequence[float] = (),
    design: OverlapDesign = DESIGN,
) -> pd.DataFrame:
    """One row per unit; out-of-target units carry outcomes that must never be used."""
    rows = []

    def add(first: int | None, second: int | None, value: float) -> None:
        a = None if first is None else (design.first.treatment if first else design.first.control)
        b = (
            None
            if second is None
            else (design.second.treatment if second else design.second.control)
        )
        rows.append(
            {
                "unit_id": f"u{len(rows):05d}",
                "elig_a": first is not None,
                "elig_b": second is not None,
                "arm_a": a,
                "arm_b": b,
                "at_a": T0 if a is not None else None,
                "at_b": T0 + dt.timedelta(hours=2) if b is not None else None,
                "start": START,
                "end": END,
                "revenue": float(value),
            }
        )

    for (first, second), values in cells.items():
        for v in values:
            add(first, second, v)
    for v in first_only:
        add(1, None, v)
    for v in second_only:
        add(None, 0, v)
    for v in neither:
        add(None, None, v)
    frame = pd.DataFrame(rows)
    for column in ("at_a", "at_b", "start", "end"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def generated_cells(
    rng: np.random.Generator,
    means: Mapping[tuple[int, int], float],
    sizes: Mapping[tuple[int, int], int],
    sd: float = 1.0,
) -> dict[tuple[int, int], list[float]]:
    return {cell: list(means[cell] + sd * rng.standard_normal(sizes[cell])) for cell in sizes}


def hc2_reference(
    frame: pd.DataFrame, weights: Sequence[float], design: OverlapDesign = DESIGN
) -> tuple[float, float]:
    """``(estimate, se)`` of ``sum_k w_k mu_k`` (cells ordered 00, 10, 01, 11) from OLS + HC2."""
    target = frame[frame["elig_a"] & frame["elig_b"]]
    a = (target["arm_a"] == design.first.treatment).to_numpy(dtype=float)
    b = (target["arm_b"] == design.second.treatment).to_numpy(dtype=float)
    y = target["revenue"].to_numpy(dtype=float)
    x = np.column_stack([np.ones_like(a), a, b, a * b])
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    meat = (x * (resid**2 / (1.0 - leverage))[:, None]).T @ x
    cov = xtx_inv @ meat @ xtx_inv
    # Cell means as linear maps of beta: mu00 = b0, mu10 = b0+bA, mu01 = b0+bB, mu11 = all.
    to_mean = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=float)
    contrast = np.asarray(weights, dtype=float) @ to_mean
    return float(contrast @ beta), float(np.sqrt(contrast @ cov @ contrast))
