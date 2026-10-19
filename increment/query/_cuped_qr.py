"""Stable factorization of a retained, scaled warehouse design relation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import ibis
import ibis.expr.types as ir
import numpy as np
from scipy.linalg import qr as scipy_qr
from scipy.linalg import solve_triangular

from increment.estimation.cuped._numeric import _fail
from increment.estimation.cuped.multicovariate import M1_MAX_LEVERAGE
from increment.query._cuped_cells import _rows


def _sum_sql(expressions: Iterable[Any]) -> Any:
    total = ibis.literal(0.0)
    for expression in expressions:
        total = total + expression
    return total


@dataclass(frozen=True, slots=True)
class QRFactor:
    """Bounded triangular factor and transformed outcome right-hand sides."""

    triangular_factor: tuple[tuple[float, ...], ...]
    transformed_rhs: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class QRUnitSummary:
    """Warehouse-resident row quantities and bounded leverage reductions."""

    relation: ir.Table
    max_leverage: float
    sum_leverage: float
    refusal_witness: tuple[tuple[str, object, float], ...]


def solve_qr_coefficients(
    factor: QRFactor,
    *,
    design_scales: Sequence[float],
    outcome_scales: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """Solve the small triangular systems and restore input column scales."""
    triangular = np.asarray(factor.triangular_factor, dtype=float)
    transformed = np.asarray(factor.transformed_rhs, dtype=float)
    design_scale = np.asarray(design_scales, dtype=float)
    outcome_scale = np.asarray(outcome_scales, dtype=float)
    solved = solve_triangular(triangular, transformed, lower=False, check_finite=False)
    coefficients = solved * outcome_scale[None, :] / design_scale[:, None]
    return tuple(
        tuple(float(value) for value in coefficients[:, component])
        for component in range(coefficients.shape[1])
    )


def warehouse_qr_unit_summary(
    relation: ir.Table,
    factor: QRFactor,
    *,
    design_columns: Sequence[str],
    outcome_columns: Sequence[str],
    sqrt_weight_column: str,
    design_scales: Sequence[float],
    coefficients: Sequence[Sequence[float]],
    execute: Callable[[ir.Table], object],
    scratch,
) -> QRUnitSummary:
    """Reconstruct leverage and regression residuals without exporting rows."""
    design_columns = tuple(design_columns)
    outcome_columns = tuple(outcome_columns)
    inverse_transpose = solve_triangular(
        np.asarray(factor.triangular_factor, dtype=float).T,
        np.eye(len(design_columns)),
        lower=True,
        check_finite=False,
    )
    q_columns = tuple(f"_qr_q_{index}" for index in range(len(design_columns)))
    q_values = {
        name: _sum_sql(
            relation[design_columns[design_index]]
            * ibis.literal(float(inverse_transpose[index, design_index]))
            for design_index in range(index + 1)
        )
        for index, name in enumerate(q_columns)
    }
    working = relation.mutate(**q_values)
    leverage = _sum_sql(working[name] * working[name] for name in q_columns)
    working = working.mutate(_qr_h=leverage)
    design_scales = tuple(float(value) for value in design_scales)
    residuals = {}
    for component, outcome_column in enumerate(outcome_columns):
        prediction = _sum_sql(
            relation[name]
            * ibis.literal(design_scales[index] * float(coefficients[component][index]))
            / relation[sqrt_weight_column]
            for index, name in enumerate(design_columns)
        )
        residuals[f"_qr_residual_{component}"] = relation[outcome_column] - prediction
    working = working.mutate(**residuals)
    evidence = scratch.materialize_next(working)
    record = _rows(
        execute(
            evidence.aggregate(
                _qr_max_leverage=evidence._qr_h.max(),
                _qr_sum_leverage=evidence._qr_h.sum(),
            )
        )
    )[0]
    maximum = float(record["_qr_max_leverage"])
    total = float(record["_qr_sum_leverage"])
    witness = (
        (("max_leverage", maximum, float(M1_MAX_LEVERAGE)),) if maximum > M1_MAX_LEVERAGE else ()
    )
    return QRUnitSummary(evidence, maximum, total, witness)


def factor_design_relation(
    relation: ir.Table,
    *,
    design_columns: Sequence[str],
    rhs_columns: Sequence[str],
    order_columns: Sequence[str],
    execute: Callable[[ir.Table], object],
    scratch,
    covariate_start: int = 0,
    covariate_names: Sequence[str] | None = None,
) -> QRFactor:
    """Householder-factor a weighted, centered, column-scaled design in SQL.

    Rows are ordered by the declared stable unit identity. Each reflector is
    computed over the current column from its diagonal row onward; scalar
    reflector products remain SQL subqueries and only the final R/RHS cross the
    client boundary.
    """
    design_columns = tuple(design_columns)
    rhs_columns = tuple(rhs_columns)
    covariate_names = tuple(covariate_names or design_columns[covariate_start:])
    row_index = ibis.row_number().over(order_by=[relation[name] for name in order_columns])
    work = relation.mutate(_qr_row=row_index).select("_qr_row", *design_columns, *rhs_columns)
    work = scratch.materialize_next(work)
    width = len(design_columns)
    all_columns = (*design_columns, *rhs_columns)

    for column_index in range(width):
        column_name = design_columns[column_index]
        column = work[column_name]
        remaining = work._qr_row >= column_index
        diagonal = work._qr_row == column_index
        max_stats = work.filter(remaining).aggregate(
            _qr_max=column.abs().max(),
            _qr_leading=column.max(where=diagonal),
        )
        scaled = work.cross_join(max_stats)
        safe_maximum = ibis.ifelse(max_stats._qr_max > 0.0, max_stats._qr_max, 1.0)
        square_stats = scaled.aggregate(
            _qr_scaled_square_sum=(
                (scaled[column_name] / safe_maximum)
                .pow(2)
                .sum(where=scaled._qr_row >= column_index)
            )
        )
        norm_stats = max_stats.cross_join(square_stats).mutate(
            _qr_norm=ibis.ifelse(
                max_stats._qr_max > 0.0,
                max_stats._qr_max * square_stats._qr_scaled_square_sum.sqrt(),
                0.0,
            )
        )
        # This is -copysign(norm, leading): both signed-zero cases select -norm.
        norm_stats = norm_stats.mutate(
            _qr_alpha=ibis.ifelse(
                norm_stats._qr_leading < 0.0,
                norm_stats._qr_norm,
                -norm_stats._qr_norm,
            )
        )
        reflected = work.cross_join(norm_stats).mutate(
            _qr_v=ibis.ifelse(
                work._qr_row < column_index,
                0.0,
                work[column_name]
                - ibis.ifelse(work._qr_row == column_index, norm_stats._qr_alpha, 0.0),
            )
        )
        reductions = reflected.aggregate(
            _qr_v2=(reflected._qr_v * reflected._qr_v).sum(),
            _qr_alpha=reflected._qr_alpha.max(),
            **{
                f"_qr_dot_{index}": (reflected._qr_v * reflected[name]).sum()
                for index, name in enumerate(all_columns)
                if index >= column_index
            },
        )
        reductions = reductions.mutate(
            _qr_beta=ibis.ifelse(reductions._qr_v2 > 0.0, 2.0 / reductions._qr_v2, 0.0)
        )
        source_columns = reflected.select("_qr_row", *all_columns, "_qr_v")
        updated = source_columns.cross_join(reductions)
        changes = {
            name: updated[name]
            - reductions._qr_beta * updated._qr_v * reductions[f"_qr_dot_{index}"]
            for index, name in enumerate(all_columns)
            if index >= column_index
        }
        work = scratch.materialize_next(updated.mutate(**changes).select("_qr_row", *all_columns))

    final_fields = {
        f"_qr_r_{row}_{column}": work[design_columns[column]].max(where=work._qr_row == row)
        for row in range(width)
        for column in range(row, width)
    }
    final_fields.update(
        {
            f"_qr_rhs_{row}_{component}": work[rhs_columns[component]].max(
                where=work._qr_row == row
            )
            for row in range(width)
            for component in range(len(rhs_columns))
        }
    )
    record = _rows(execute(work.aggregate(**final_fields)))[0]
    factor = tuple(
        tuple(
            float(record[f"_qr_r_{row}_{column}"]) if column >= row else 0.0
            for column in range(width)
        )
        for row in range(width)
    )
    rhs = tuple(
        tuple(float(record[f"_qr_rhs_{row}_{component}"]) for component in range(len(rhs_columns)))
        for row in range(width)
    )
    _apply_pivoted_r_rank(factor, covariate_start, covariate_names)
    return QRFactor(factor, rhs)


def _apply_pivoted_r_rank(
    factor: tuple[tuple[float, ...], ...],
    start: int,
    names: tuple[str, ...],
) -> None:
    """Apply the local fitter's column-scaled pivoted-R rank threshold."""
    if not names:
        return
    block = np.asarray(factor, dtype=float)[start:, start:]
    scales = np.linalg.norm(block, axis=0)
    if not np.isfinite(scales).all() or np.any(scales == 0.0):
        deficient = tuple(names[index] for index, scale in enumerate(scales) if scale == 0.0)
        _fail(
            "estimation.cuped_linear.rank_deficiency",
            columns=deficient or names,
            partners=(),
        )
    scaled = block / scales[None, :]
    _, pivoted, pivots = scipy_qr(scaled, mode="economic", pivoting=True, check_finite=False)
    diagonal_squares = np.diag(pivoted) ** 2
    rank = int(np.count_nonzero(diagonal_squares > len(names) * np.finfo(float).eps))
    if rank < len(names):
        _fail(
            "estimation.cuped_linear.rank_deficiency",
            columns=(names[int(pivots[rank])],),
            partners=tuple(names[int(index)] for index in pivots[:rank]),
        )
